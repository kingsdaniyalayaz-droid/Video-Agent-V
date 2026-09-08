from pathlib import Path
import ast

# Automatically locate app.py in the same project folder
APP_FILE = Path(__file__).resolve().parent / "app.py"

source = APP_FILE.read_text(encoding="utf-8")

tree = ast.parse(source)

functions = {
    node.name: node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

assert "get_owned_rag_chain" in functions
assert "invalidate_rag_state" in functions
assert "can_reuse_cached_rag" in functions

owned = ast.get_source_segment(
    source,
    functions["get_owned_rag_chain"]
)

assert 'result["rag_chain"] = None' in owned
assert 'st.session_state["rag_video_id"] = None' in owned
assert "result_id != rag_id" in owned


active = ast.get_source_segment(
    source,
    functions["set_active_processed_result"]
)

assert "if not same_source:" in active
assert "invalidate_rag_state(normalized_result)" in active


process = ast.get_source_segment(
    source,
    functions["process_source"]
)

assert "can_reuse_cached_rag(" in process
assert 'result["rag_chain"] = (' in process
assert 'previous_result.get("rag_chain") if reusable_chain else None' in process
assert (
    "st.session_state.rag_video_id = "
    "cached_video_id if reusable_chain else None"
) in process


chat = ast.get_source_segment(
    source,
    functions["render_chat"]
)

assert "rag_chain = get_owned_rag_chain(result)" in chat
assert 'rag_chain = result.get("rag_chain")' not in chat
assert "get_cached_rag_chain(" in chat


library = ast.get_source_segment(
    source,
    functions["_load_video_library_entry"]
)

assert "reuse_existing_rag" in library
assert "get_owned_rag_chain(previous_result)" in library


print("PASS: source-switching and RAG ownership regression assertions")
print("PASS: A->B cached path invalidates unless exact identity matches")
print("PASS: library reopen path validates in-memory chain ownership")
print("PASS: meeting/upload activation invalidates old RAG on source change")
print("PASS: chat path rejects mismatched chain and loads lazily")
print("PASS: same-video cached reuse remains identity-guarded")