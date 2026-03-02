import os
import psutil


class Monitor:
    def __init__(self, rank: int):
        self.rank = rank
        self.memory_records: list = []  # 存 (stage_name, memory_gb)
        self.time_records: list = []    # 存 (stage_name, duration)

        # 初始记录
        self.record_memory("initial")

    def record_memory(self, stage_name: str) -> None:
        """记录当前内存使用"""
        usage = get_memory_usage()
        self.memory_records.append((stage_name, usage))

    def record_time(self, stage_name: str, duration: float) -> None:
        """记录计算耗时"""
        self.time_records.append((stage_name, duration))

    def save_with_total_time(self, path: str, total_time: float, comm, root: int = 0) -> None:
        """
        收集所有 Rank 的数据并在 Root 节点保存文件
        """
        # Gather data
        all_memory = comm.gather(self.memory_records, root=root)
        all_times = comm.gather(self.time_records, root=root)

        if self.rank == root:
            write_timing_status(path, all_times, total_time)
            write_memory_status(path, all_memory)
            print("Timing and Memory statistics written.")


def get_memory_usage() -> float:
    """Get current memory usage of the process in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB


def write_memory_status(path: str, all_memory: list) -> None:
    """Write memory usage statistics to a file.

    Note: values are in GB (despite the header saying MB in the original code).
    We preserve the original output format for backward compatibility.
    """
    with open(os.path.join(path, "memory_status.txt"), "w") as f:
        f.write("Memory Usage Statistics (MB)\n")
        f.write("=" * 50 + "\n\n")

        for rank_idx, rank_memory in enumerate(all_memory):
            f.write(f"Rank {rank_idx} Memory Usage:\n")
            f.write("-" * 30 + "\n")
            for stage, memory in rank_memory:
                f.write(f"{stage}: {memory:.2f} MB\n")
            f.write("\n")

            # Calculate memory differences
            if len(rank_memory) > 1:
                f.write("Memory Changes:\n")
                for i in range(1, len(rank_memory)):
                    prev_stage, prev_memory = rank_memory[i-1]
                    curr_stage, curr_memory = rank_memory[i]
                    diff = curr_memory - prev_memory
                    f.write(f"{prev_stage} -> {curr_stage}: {diff:+.2f} MB\n")
            f.write("\n")


def write_timing_status(path: str, all_times: list, total_time: float) -> None:
    """Write timing statistics to a file"""
    with open(os.path.join(path, "timing_status.txt"), "w") as f:
        f.write("Timing Statistics\n")
        f.write("=" * 50 + "\n\n")

        # Write total execution time
        f.write(f"Total execution time: {total_time:.6f} seconds\n\n")

        # Write per-rank timing information
        for rank_idx, rank_times in enumerate(all_times):
            f.write(f"Rank {rank_idx} Timing:\n")
            f.write("-" * 30 + "\n")
            for stage, time in rank_times:
                f.write(f"{stage}: {time:.6f} seconds\n")
            f.write("\n")
