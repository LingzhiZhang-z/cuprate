"""Canonical spin-operator keys and JSON serialization helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from cuprate import ATOL

OperatorKey = tuple[tuple[int, int], ...]


def canonical_operator_key(sites: Sequence[int]) -> OperatorKey:
    """Return the canonical pair-list key for one spin-operator term."""
    if len(sites) % 2 != 0:
        raise ValueError("operator term must contain an even number of sites")
    pairs = []
    for offset in range(0, len(sites), 2):
        i, j = int(sites[offset]), int(sites[offset + 1])
        pairs.append((min(i, j), max(i, j)))
    return tuple(sorted(pairs))


def key_to_json(key: OperatorKey) -> list[list[int]]:
    return [[int(i), int(j)] for i, j in key]


def key_from_json(value: Sequence[Sequence[int]]) -> OperatorKey:
    return canonical_operator_key([site for pair in value for site in pair])


def key_to_sites(key: OperatorKey) -> list[int]:
    return [site for pair in key for site in pair]


def map_key(key: OperatorKey, index_mapping: dict[int, int]) -> OperatorKey:
    sites = []
    for pair in key:
        sites.extend(index_mapping[int(site)] for site in pair)
    return canonical_operator_key(sites)


def complex_json(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def complex_from_json(value: dict[str, Any]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def operators_json(
    cluster_sites: Sequence[Sequence[int]],
    bond_groups: list[list[Sequence[int]]],
    coeffs: list[Any],
) -> dict[str, Any]:
    constant, terms = terms_from_fit(bond_groups, coeffs)
    return operators_from_terms(cluster_sites, constant, terms)


def terms_from_fit(
    bond_groups: list[list[Sequence[int]]],
    coeffs: list[Any],
) -> tuple[complex, dict[OperatorKey, complex]]:
    if len(coeffs) != len(bond_groups) + 1:
        raise ValueError("coeffs must contain one constant plus one coefficient group per bond group")
    terms: dict[OperatorKey, complex] = {}
    for group_idx, (group, group_coeffs) in enumerate(zip(bond_groups, coeffs[1:])):
        if len(group_coeffs) != len(group):
            raise ValueError(
                f"coefficient group {group_idx} has {len(group_coeffs)} values "
                f"for {len(group)} operator terms"
            )
        for term, coefficient in zip(group, group_coeffs):
            terms[canonical_operator_key(term)] = complex(coefficient)
    return complex(coeffs[0]), terms


def operators_from_terms(
    cluster_sites: Sequence[Sequence[int]],
    constant: complex,
    terms: dict[OperatorKey, complex],
    *,
    drop_small: bool = False,
) -> dict[str, Any]:
    two_site: dict[tuple[int, int], list[tuple[OperatorKey, complex]]] = defaultdict(list)
    multi_site: dict[int, dict[tuple[int, ...], list[tuple[OperatorKey, complex]]]] = {
        4: defaultdict(list),
        6: defaultdict(list),
    }

    for key, coefficient in terms.items():
        if drop_small and abs(coefficient) <= ATOL["tight"]:
            continue
        arity = 2 * len(key)
        if arity == 2:
            two_site[_bond_vector(cluster_sites, key)].append((key, coefficient))
        elif arity in multi_site:
            support = tuple(sorted({site for pair in key for site in pair}))
            multi_site[arity][support].append((key, coefficient))
        else:
            raise ValueError(f"unsupported operator arity: {arity}")

    groups: list[dict[str, Any]] = []
    for label_idx, vector in enumerate(sorted(two_site, key=_vector_sort_key), start=1):
        term_items = sorted(two_site[vector], key=lambda item: _support_sort_key(cluster_sites, item[0]))
        groups.append(
            {
                "arity": 2,
                "vector": [int(vector[0]), int(vector[1])],
                "label": f"J{label_idx}",
                "terms": [_term_json(key, coefficient) for key, coefficient in term_items],
            }
        )

    for arity, prefix in ((4, "K"), (6, "L")):
        label_idx = 1
        supports = sorted(multi_site[arity], key=lambda support: _support_sort_key(cluster_sites, support))
        for support in supports:
            term_items = sorted(
                multi_site[arity][support],
                key=lambda item: _term_sort_key(cluster_sites, item[0]),
            )
            groups.append(
                {
                    "arity": arity,
                    "vector": None,
                    "label": f"{prefix}{label_idx}",
                    "support": [int(site) for site in support],
                    "terms": [_term_json(key, coefficient) for key, coefficient in term_items],
                }
            )
            label_idx += 1

    return {
        "constant_term": complex_json(constant),
        "groups": groups,
    }


def operators_to_terms(payload: dict[str, Any]) -> tuple[complex, dict[OperatorKey, complex]]:
    constant = complex_from_json(payload["constant_term"])
    terms: dict[OperatorKey, complex] = {}
    for group in payload["groups"]:
        for term in group["terms"]:
            if "key" not in term:
                raise ValueError("operator term is missing canonical key")
            key = key_from_json(term["key"])
            terms[key] = complex_from_json(term["coefficient"])
    return constant, terms


def operator_summary(constant: complex, terms: dict[OperatorKey, complex]) -> dict[str, Any]:
    magnitudes = [abs(value) for value in terms.values()]
    max_term = max(magnitudes) if magnitudes else 0.0
    return {
        "constant_term": complex_json(constant),
        "term_count": int(len(terms)),
        "max_abs_coefficient": float(max(abs(constant), max_term)),
    }


def max_operator_difference(
    left_constant: complex,
    left_terms: dict[OperatorKey, complex],
    right_constant: complex,
    right_terms: dict[OperatorKey, complex],
) -> float:
    keys = set(left_terms) | set(right_terms)
    term_error = max(
        (abs(left_terms.get(key, 0.0) - right_terms.get(key, 0.0)) for key in keys),
        default=0.0,
    )
    return float(max(abs(left_constant - right_constant), term_error))


def _term_json(key: OperatorKey, coefficient: complex) -> dict[str, Any]:
    return {
        "sites": key_to_sites(key),
        "key": key_to_json(key),
        "coefficient": complex_json(coefficient),
    }


def _bond_vector(cluster_sites: Sequence[Sequence[int]], key: OperatorKey) -> tuple[int, int]:
    if len(key) != 1:
        raise ValueError("bond vector is defined only for two-site operators")
    site1, site2 = key[0]
    dx = abs(int(cluster_sites[site2][0]) - int(cluster_sites[site1][0]))
    dy = abs(int(cluster_sites[site2][1]) - int(cluster_sites[site1][1]))
    return tuple(sorted((dx, dy), reverse=True))


def _vector_sort_key(vector: tuple[int, int]) -> tuple[int, int, int]:
    dx, dy = vector
    return (dx * dx + dy * dy, dx, dy)


def _support_sort_key(
    cluster_sites: Sequence[Sequence[int]],
    support_or_key: Sequence[int] | OperatorKey,
) -> tuple[tuple[int, int, int], ...]:
    if support_or_key and isinstance(support_or_key[0], tuple):
        support = sorted({site for pair in support_or_key for site in pair})  # type: ignore[union-attr]
    else:
        support = [int(site) for site in support_or_key]  # type: ignore[arg-type]
    return tuple(
        (int(cluster_sites[site][0]), int(cluster_sites[site][1]), int(site))
        for site in support
    )


def _term_sort_key(
    cluster_sites: Sequence[Sequence[int]],
    key: OperatorKey,
) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(sorted((_pair_descriptor(cluster_sites, pair) for pair in key)))


def _pair_descriptor(
    cluster_sites: Sequence[Sequence[int]],
    pair: tuple[int, int],
) -> tuple[int, int, int, int, int]:
    i, j = pair
    if i > j:
        i, j = j, i
    dx = int(cluster_sites[j][0]) - int(cluster_sites[i][0])
    dy = int(cluster_sites[j][1]) - int(cluster_sites[i][1])
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return (dx * dx + dy * dy, dx, dy, i, j)
