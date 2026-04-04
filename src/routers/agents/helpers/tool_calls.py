"""
Tool call formatting helpers for agent completion.

Handles formatting tool call data according to the chat_message.v2.json schema.
"""
from typing import Dict, Any, Optional, List
import ast
import json as _json


def format_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any
) -> Optional[Dict[str, Any]]:
    """
    Format a tool call according to the chat_message.v2.json schema.
    
    Determines the tool type from the tool name and formats the input/output
    appropriately for each tool type.
    
    Args:
        tool_name: The name of the tool that was executed
        tool_input: The input that was passed to the tool
        tool_output: The output returned by the tool
        
    Returns:
        Formatted tool call dict, or None if the tool output is invalid
    """
    # Normalize tool_output to a dict.
    # LangChain's StructuredTool.arun() calls str() on the tool's return value before
    # passing it to the on_tool_end callback, so tool_output is typically a Python repr
    # string of the original dict (e.g. "{'results': [...], 'namespace': '...'}").
    # Newer LangChain versions may also pass a ToolMessage/AIMessage object — extract
    # the string content from those before attempting to parse.
    if not isinstance(tool_output, dict):
        # Unwrap LangChain message objects (ToolMessage, AIMessage, etc.)
        if hasattr(tool_output, "content"):
            tool_output = tool_output.content

        if isinstance(tool_output, str) and tool_output.strip():
            # Try JSON first (clean format), then Python literal eval (str() format)
            try:
                parsed = _json.loads(tool_output)
                tool_output = parsed if isinstance(parsed, dict) else {"result": tool_output}
            except (_json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(tool_output)
                    tool_output = parsed if isinstance(parsed, dict) else {"result": tool_output}
                except (ValueError, SyntaxError):
                    print(f"[TOOL CALLS] Could not parse tool_output string; wrapping as result")
                    tool_output = {"result": tool_output}
        else:
            print(f"[TOOL CALLS] Normalizing non-dict/non-str tool_output (type: {type(tool_output).__name__})")
            tool_output = {"result": str(tool_output)}
    
    # Dispatch by fixed tool name
    _FORMATTERS = {
        "vector_search": _format_vector_search,
        "vector_search_with_reranking": _format_vector_search_rerank,
        "db_table_read": _format_db_table_read,
        "db_table_write": _format_db_table_write,
        "send_txt_email_with_ses": _format_send_txt_email,
        "send_html_email_with_ses": _format_send_html_email,
        "send_txt_email_with_google_oauth": _format_send_txt_email,
        "send_txt_email_with_google_smtp": _format_send_txt_email,
    }
    formatter = _FORMATTERS.get(tool_name, _format_generic_tool)
    return formatter(tool_name, tool_input, tool_output)


def _format_vector_search(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Format vector search tool call. Returns None for error outputs."""
    if 'error' in tool_output and 'results' not in tool_output:
        print(f"[TOOL CALLS] Skipping failed vector_search call: {tool_output.get('error', '')[:100]}")
        return None

    results = _format_search_results(tool_output.get('results', []))

    input_data: Dict[str, Any] = {"query": tool_input.get('query', '')}
    top_k = tool_input.get('top_k', tool_input.get('topK'))
    if top_k is not None:
        input_data["topK"] = int(top_k)

    return {
        "toolType": "vectorSearch",
        "toolName": tool_name,
        "input": input_data,
        "output": {
            "results": results,
            "namespace": tool_output.get('namespace', ''),
            "index": tool_output.get('index', '')
        }
    }


def _format_vector_search_rerank(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Format vector search with reranking tool call. Returns None for error outputs."""
    if 'error' in tool_output and 'results' not in tool_output:
        print(f"[TOOL CALLS] Skipping failed vector_search_rerank call: {tool_output.get('error', '')[:100]}")
        return None

    results = _format_search_results(tool_output.get('results', []))

    input_data: Dict[str, Any] = {"query": tool_input.get('query', '')}
    top_k = tool_input.get('top_k', tool_input.get('topK'))
    if top_k is not None:
        input_data["topK"] = int(top_k)

    return {
        "toolType": "vectorSearchWithReranking",
        "toolName": tool_name,
        "input": input_data,
        "output": {
            "results": results,
            "namespace": tool_output.get('namespace', ''),
            "index": tool_output.get('index', '')
        }
    }


def _format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format search results according to v2 schema."""
    formatted = []
    for result in results:
        formatted.append({
            "id": result.get("id", ""),
            "score": result.get("score", 0.0),
            "metadata": result.get("metadata", {})
        })
    return formatted


def _format_db_table_read(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Format database table read tool call."""
    return {
        "toolType": "dbTableRead",
        "toolName": tool_name,
        "input": {
            "filters": tool_input.get('filters'),
            "limit": tool_input.get('limit'),
            "offset": tool_input.get('offset')
        },
        "output": {
            "results": tool_output.get('results', []),
            "table": tool_output.get('table', ''),
            "count": tool_output.get('count', 0)
        }
    }


def _format_db_table_write(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Format database table write tool call."""
    return {
        "toolType": "dbTableWrite",
        "toolName": tool_name,
        "input": {
            "data": tool_input  # The flat input IS the data
        },
        "output": {
            "success": tool_output.get('success', False),
            "table": tool_output.get('table', ''),
            "inserted": tool_output.get('inserted', {}),
            "message": tool_output.get('message', ''),
            "error": tool_output.get('error')
        }
    }


def _format_send_txt_email(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Format a send-plain-text-email tool call."""
    return {
        "toolType": "sendTxtEmailWithSes",
        "toolName": tool_name,
        "input": {
            "to": tool_input.get("to_email", tool_input.get("to", "")),
            "subject": tool_input.get("subject", ""),
            "body": tool_input.get("body", ""),
        },
        "output": {
            "success": tool_output.get("success", False),
            "messageId": tool_output.get("message_id", tool_output.get("messageId")),
            "error": tool_output.get("error"),
        },
    }


def _format_send_html_email(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any],
) -> Dict[str, Any]:
    """Format a send-HTML-email tool call."""
    return {
        "toolType": "sendHtmlEmailWithSes",
        "toolName": tool_name,
        "input": {
            "to": tool_input.get("to_email", tool_input.get("to", "")),
            "subject": tool_input.get("subject", ""),
            "html_body": tool_input.get("html_body", ""),
        },
        "output": {
            "success": tool_output.get("success", False),
            "messageId": tool_output.get("message_id", tool_output.get("messageId")),
            "error": tool_output.get("error"),
        },
    }


def _format_generic_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Dict[str, Any]
) -> Dict[str, Any]:
    """Format generic/unknown tool call."""
    return {
        "toolType": "custom",
        "toolName": tool_name,
        "input": tool_input,
        "output": tool_output
    }
