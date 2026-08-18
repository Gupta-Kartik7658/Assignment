"""
Quick sanity checks (plain asserts, run with: py test_tree.py).
"""

import summary_tree
from summary_tree import MAX_DEPTH, Tree


def test_depth_cap():
    tree = Tree(word_threshold=1000, time_threshold=1000)
    # keep drilling down; the tree must stop growing at MAX_DEPTH
    for i in range(10):
        tree.insert_chunk(f"chunk {i}", hint="child")
    assert tree.active.depth == MAX_DEPTH, tree.active.depth

    depths = [n.depth for n in tree._all_nodes()]
    assert max(depths) == MAX_DEPTH

    # the extra chunks piled onto the deepest node instead of new levels
    deepest = tree.active
    assert len(deepest.raw_chunks) == 10 - MAX_DEPTH + 1, deepest.raw_chunks
    print("depth cap ok")


def test_one_call_per_chunk_at_most():
    tree = Tree(word_threshold=4, time_threshold=1000)
    before = summary_tree.llm_call_count
    for i in range(5):
        tree.insert_chunk("one two three four five")
    calls = summary_tree.llm_call_count - before
    assert calls <= 5, calls
    print(f"per-chunk budget ok ({calls} calls for 5 chunks)")


def test_no_rollup_during_chunks():
    tree = Tree(word_threshold=4, time_threshold=1000)
    tree.insert_chunk("payments migration is going fine today")
    tree.insert_chunk("next topic mobile release", hint="child")
    # root summary was never touched, only marked dirty
    assert tree.root.dirty is True
    assert tree.root.subtree_summary == ""

    n = tree.rollup_dirty_nodes()
    assert n >= 1
    assert tree.root.dirty is False
    assert tree.root.subtree_summary != ""
    print(f"lazy rollup ok ({n} nodes rolled up)")


if __name__ == "__main__":
    test_depth_cap()
    test_one_call_per_chunk_at_most()
    test_no_rollup_during_chunks()
    print("all good")
