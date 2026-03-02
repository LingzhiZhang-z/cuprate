from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Params:
    # 物理参数
    N: int = 3
    U: float = 1.0
    t: float = 0.1
    t2: Optional[float] = None
    t3: Optional[float] = None

    # 自旋参数
    sz: Optional[float] = None
    s2: Optional[float] = None
    s2_fix: bool = False
    s: Optional[float] = None

    # 计算类型
    type: Optional[str] = None
    block: Optional[str] = None
    restart: bool = False

    # 绝热参数
    delta: Optional[float] = None
    delta2: Optional[float] = None
    type_delta: Optional[str] = None

    # 嵌入参数（仅 embed 使用，可考虑移除）
    Ncell: Optional[int] = None
    Ncut: Optional[int] = None
    ratio: Optional[float] = None

    # 运行时注入（不通过 CLI）
    path_spec: Optional[object] = field(default=None, repr=False)
    result_dir: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Convert to dict for backward compatibility with code that expects dict params."""
        return {
            'N': self.N, 'U': self.U, 't': self.t, 't2': self.t2, 't3': self.t3,
            'sz': self.sz, 's2': self.s2, 's2_fix': self.s2_fix, 's': self.s,
            'type': self.type, 'block': self.block, 'restart': self.restart,
            'delta': self.delta, 'delta2': self.delta2, 'type_delta': self.type_delta,
            'Ncell': self.Ncell, 'Ncut': self.Ncut, 'ratio': self.ratio,
            'path_spec': self.path_spec, 'result_dir': self.result_dir,
        }

    def __getitem__(self, key: str):
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        """Allow dict-style assignment for backward compatibility."""
        setattr(self, key, value)

    def get(self, key: str, default=None):
        """Allow dict-style .get() for backward compatibility."""
        return getattr(self, key, default)
