"""Write the code excerpts of the note as LaTeX listings, located in the repository files by text anchors."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent


def excerpt(rel_path: str, start_anchor: str, end_anchor: str, name: str, needs: int = 10) -> None:
    """Write ``name.tex`` holding the lines of ``rel_path`` from the first line containing ``start_anchor`` to the
    first later line containing ``end_anchor``, with the real line numbers in the header."""
    lines = (ROOT / rel_path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if start_anchor in l)
    end = next(i for i in range(start, len(lines)) if end_anchor in lines[i])
    body = "\n".join(lines[start : end + 1])
    tex = (
        f"\\begin{{snippetbox}}[{needs}]\n"
        f"\\snippethead{{{rel_path}}}{{lines {start + 1}--{end + 1}}}\n"
        f"\\begin{{lstlisting}}[firstnumber={start + 1}]\n{body}\n\\end{{lstlisting}}\n"
        f"\\end{{snippetbox}}\n"
    )
    (OUT / f"{name}.tex").write_text(tex)
    print(name, start + 1, end + 1)


MDP = "source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset/mdp"
excerpt(f"{MDP}/rigid_object_hasher.py", "coll_prims = get_all_matching_child_prims(", "if len(coll_prims) == 0:", "snip_callsite", 12)
excerpt(f"{MDP}/rigid_object_hasher.py", "@staticmethod", "return own", "snip_own_colliders", 14)
excerpt(f"{MDP}/utils.py", "mp = wp.mesh_query_point(mid, q3, max_dist)", "signs[tid] = best_signed_dist", "snip_kernel", 12)
excerpt(f"{MDP}/collision_analyzer.py", "signs = (", "return coll_free_mask", "snip_verdict", 8)
