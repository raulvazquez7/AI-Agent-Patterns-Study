"""Implement A2A (Agent2Agent) endpoint for JSON-RPC 2.0 protocol.

The Agent2Agent (A2A) Protocol is an open standard designed to facilitate
communication and interoperability between independent AI agent systems.

A2A Protocol specification:
https://a2a-protocol.org/dev/specification/

The implementation currently supports JSON-RPC 2.0 transport only.
Streaming (SSE) and push notifications are not implemented.
"""

import functools
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, NotRequired, cast

import orjson
from langgraph_sdk.client import LangGraphClient, get_client
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from structlog import getLogger
from typing_extensions import TypedDict

from langgraph_api.metadata import USER_API_URL
from langgraph_api.route import ApiRequest, ApiRoute
from langgraph_api.utils.cache import LRUCache

logger = getLogger(__name__)

# Cache for assistant schemas (assistant_id -> schemas dict)
_assistant_schemas_cache = LRUCache[dict[str, Any]](max_size=1000, ttl=60)


# ============================================================================
# JSON-RPC 2.0 Base Types (shared with MCP)
# ============================================================================


class JsonRpcErrorObject(TypedDict):
    code: int
    message: str
    data: NotRequired[Any]


class JsonRpcRequest(TypedDict):
    jsonrpc: Literal["2.0"]
    id: str | int
    method: str
    params: NotRequired[dict[str, Any]]


class JsonRpcResponse(TypedDict):
    jsonrpc: Literal["2.0"]
    id: str | int
    result: NotRequired[dict[str, Any]]
    error: NotRequired[JsonRpcErrorObject]


class JsonRpcNotification(TypedDict):
    jsonrpc: Literal["2.0"]
    method: str
    params: NotRequired[dict[str, Any]]


# ============================================================================
# A2A Specific Error Codes
# ============================================================================

# Standard JSON-RPC error codes
ERROR_CODE_PARSE_ERROR = -32700
ERROR_CODE_INVALID_REQUEST = -32600
ERROR_CODE_METHOD_NOT_FOUND = -32601
ERROR_CODE_INVALID_PARAMS = -32602
ERROR_CODE_INTERNAL_ERROR = -32603

# A2A-specific error codes (in server error range -32000 to -32099)
ERROR_CODE_TASK_NOT_FOUND = -32001
ERROR_CODE_TASK_NOT_CANCELABLE = -32002
ERROR_CODE_PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
ERROR_CODE_UNSUPPORTED_OPERATION = -32004
ERROR_CODE_CONTENT_TYPE_NOT_SUPPORTED = -32005
ERROR_CODE_INVALID_AGENT_RESPONSE = -32006


# ============================================================================
# Constants and Configuration
# ============================================================================

A2A_PROTOCOL_VERSION = "0.3.0"


@functools.lru_cache(maxsize=1)
def _client() -> LangGraphClient:
    """Get a client for local operations."""
    return get_client(url=None)


def _get_version() -> str:
    """Get langgraph-api version."""
    from langgraph_api import __version__

    return __version__


def _generate_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())


def _generate_context_id() -> str:
    """Generate a unique context ID."""
    return str(uuid.uuid4())


def _generate_timestamp() -> str:
    """Generate ISO 8601 timestamp."""
    return datetime.now(UTC).isoformat()


async def _get_assistant(
    client: LangGraphClient, assistant_id: str, headers: Headers | dict[str, Any] | None
) -> dict[str, Any]:
    """Get assistant with proper 404 error handling.

    Args:
        client: LangGraph client
        assistant_id: The assistant ID to get
        headers: Request headers

    Returns:
        The assistant dictionary

    Raises:
        ValueError: If assistant not found or other errors
    """
    try:
        return await client.assistants.get(assistant_id, headers=headers)
    except Exception as e:
        if (
            hasattr(e, "response")
            and hasattr(e.response, "status_code")
            and e.response.status_code == 404
        ):
            raise ValueError(f"Assistant '{assistant_id}' not found") from e
        raise ValueError(f"Failed to get assistant '{assistant_id}': {e}") from e


async def _validate_supports_messages(
    client: LangGraphClient,
    assistant: dict[str, Any],
    headers: Headers | dict[str, Any] | None,
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that assistant supports messages if text parts are present.

    If the parts contain text parts, the agent must support the 'messages' field.
    If the parts only contain data parts, no validation is performed.

    Args:
        client: LangGraph client
        assistant: The assistant dictionary
        headers: Request headers
        parts: The original A2A message parts

    Returns:
        The schemas dictionary from the assistant

    Raises:
        ValueError: If assistant doesn't support messages when text parts are present
    """
    assistant_id = assistant["assistant_id"]

    cached_schemas = await _assistant_schemas_cache.get(assistant_id)
    if cached_schemas is not None:
        schemas = cached_schemas
    else:
        try:
            schemas = await client.assistants.get_schemas(assistant_id, headers=headers)
            _assistant_schemas_cache.set(assistant_id, schemas)
        except Exception as e:
            raise ValueError(
                f"Failed to get schemas for assistant '{assistant_id}': {e}"
            ) from e

    # Validate messages field only if there are text parts
    has_text_parts = any(part.get("kind") == "text" for part in parts)
    if has_text_parts:
        input_schema = schemas.get("input_schema")
        if not input_schema:
            raise ValueError(
                f"Assistant '{assistant_id}' has no input schema defined. "
                f"A2A conversational agents using text parts must have an input schema with a 'messages' field."
            )

        properties = input_schema.get("properties", {})
        if "messages" not in properties:
            graph_id = assistant["graph_id"]
            raise ValueError(
                f"Assistant '{assistant_id}' (graph '{graph_id}') does not support A2A conversational messages. "
                f"Graph input schema must include a 'messages' field to accept text parts. "
                f"Available input fields: {list(properties.keys())}"
            )

    return schemas


def _process_a2a_message_parts(
    parts: list[dict[str, Any]], message_role: str
) -> dict[str, Any]:
    """Convert A2A message parts to LangChain messages format.

    Args:
        parts: List of A2A message parts
        message_role: A2A message role ("user" or "agent")

    Returns:
        Input content with messages in LangChain format

    Raises:
        ValueError: If message parts are invalid
    """
    messages = []
    additional_data = {}

    for part in parts:
        part_kind = part.get("kind")

        if part_kind == "text":
            # Text parts become messages with role based on A2A message role
            if "text" not in part:
                raise ValueError("TextPart must contain a 'text' field")

            # Map A2A role to LangGraph role
            langgraph_role = "human" if message_role == "user" else "assistant"
            messages.append({"role": langgraph_role, "content": part["text"]})

        elif part_kind == "data":
            # Data parts become structured input parameters
            part_data = part.get("data", {})
            if not isinstance(part_data, dict):
                raise ValueError(
                    "DataPart must contain a JSON object in the 'data' field"
                )
            additional_data.update(part_data)

        else:
            raise ValueError(
                f"Unsupported part kind '{part_kind}'. "
                f"A2A agents support 'text' and 'data' parts only."
            )

    if not messages and not additional_data:
        raise ValueError("Message must contain at least one valid text or data part")

    # Create input with messages in LangChain format
    input_content = {}
    if messages:
        input_content["messages"] = messages
    if additional_data:
        input_content.update(additional_data)

    return input_content


def _extract_a2a_response(result: dict[str, Any]) -> str:
    """Extract the last assistant message from graph execution result.

    Args:
        result: Graph execution result

    Returns:
        Content of the last assistant message

    Raises:
        ValueError: If result doesn't contain messages or is invalid
    """
    if "__error__" in result:
        # Let the caller handle errors
        return str(result)

    if "messages" not in result:
        # Fallback to the full result if no messages schema. It is not optimal to do A2A on assistants without
        # a messages key, but it is not a hard requirement.
        return str(result)

    messages = result["messages"]
    if not isinstance(messages, list) or not messages:
        return str(result)

    # Find the last assistant message
    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and "content" in message
            or message.get("type") == "ai"
            and "content" in message
        ):
            return message["content"]

    # If no assistant message found, return the last message content
    last_message = messages[-1]
    if isinstance(last_message, dict):
        return last_message.get("content", str(last_message))

    return str(last_message)


def _map_runs_create_error_to_rpc(
    exception: Exception, assistant_id: str, thread_id: str | None = None
) -> dict[str, Any]:
    """Map runs.create() exceptions to A2A JSON-RPC error responses.

    Args:
        exception: Exception from runs.create()
        assistant_id: The assistant ID that was used
        thread_id: The thread ID that was used (if any)

    Returns:
        A2A error response dictionary
    """
    if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        status_code = exception.response.status_code
        error_text = str(exception)

        if status_code == 404:
            # Check if it's a thread or assistant not found
            if "thread" in error_text.lower() or "Thread" in error_text:
                return {
                    "error": {
                        "code": ERROR_CODE_INVALID_PARAMS,
                        "message": f"Thread '{thread_id}' not found. Please create the thread first before sending messages to it.",
                        "data": {
                            "thread_id": thread_id,
                            "error_type": "thread_not_found",
                        },
                    }
                }
            else:
                return {
                    "error": {
                        "code": ERROR_CODE_INVALID_PARAMS,
                        "message": f"Assistant '{assistant_id}' not found",
                    }
                }
        elif status_code == 400:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": f"Invalid request: {error_text}",
                }
            }
        elif status_code == 403:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Access denied to assistant or thread",
                }
            }
        else:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": f"Failed to create run: {error_text}",
                }
            }

    return {
        "error": {
            "code": ERROR_CODE_INTERNAL_ERROR,
            "message": f"Internal server error: {str(exception)}",
        }
    }


def _map_runs_get_error_to_rpc(
    exception: Exception, task_id: str, thread_id: str
) -> dict[str, Any]:
    """Map runs.get() exceptions to A2A JSON-RPC error responses.

    Args:
        exception: Exception from runs.get()
        task_id: The task/run ID that was requested
        thread_id: The thread ID that was requested

    Returns:
        A2A error response dictionary
    """
    if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        status_code = exception.response.status_code
        error_text = str(exception)

        status_code_handlers = {
            404: {
                "error": {
                    "code": ERROR_CODE_TASK_NOT_FOUND,
                    "message": f"Task '{task_id}' not found in thread '{thread_id}'",
                }
            },
            400: {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": f"Invalid request: {error_text}",
                }
            },
            403: {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Access denied to task",
                }
            },
        }

        return status_code_handlers.get(
            status_code,
            {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": f"Failed to get task: {error_text}",
                }
            },
        )

    return {
        "error": {
            "code": ERROR_CODE_INTERNAL_ERROR,
            "message": f"Internal server error: {str(exception)}",
        }
    }


def _create_task_response(
    task_id: str,
    context_id: str,
    message: dict[str, Any],
    result: dict[str, Any],
    assistant_id: str,
) -> dict[str, Any]:
    """Create A2A Task response structure for both success and failure cases.

    Args:
        task_id: The task/run ID
        context_id: The context/thread ID
        message: Original A2A message from request
        result: LangGraph execution result
        assistant_id: The assistant ID used

    Returns:
        A2A Task response dictionary
    """
    base_task = {
        "id": task_id,
        "contextId": context_id,
        "history": [
            {**message, "taskId": task_id, "contextId": context_id, "kind": "message"}
        ],
        "kind": "task",
    }

    if "__error__" in result:
        base_task["status"] = {
            "state": "failed",
            "message": {
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"Error executing assistant: {result['__error__']['error']}",
                    }
                ],
                "messageId": _generate_task_id(),
                "taskId": task_id,
                "contextId": context_id,
                "kind": "message",
            },
        }
    else:
        artifact_id = _generate_task_id()
        artifacts = [
            {
                "artifactId": artifact_id,
                "name": "Assistant Response",
                "description": f"Response from assistant {assistant_id}",
                "parts": [
                    {
                        "kind": "text",
                        "text": _extract_a2a_response(result),
                    }
                ],
            }
        ]

        base_task["status"] = {
            "state": "completed",
            "timestamp": _generate_timestamp(),
        }
        base_task["artifacts"] = artifacts

    return {"result": base_task}


# ============================================================================
# Main A2A Endpoint Handler
# ============================================================================


def handle_get_request() -> Response:
    """Handle HTTP GET requests (streaming not currently supported).

    Returns:
        405 Method Not Allowed
    """
    return Response(status_code=405)


def handle_delete_request() -> Response:
    """Handle HTTP DELETE requests (session termination not currently supported).

    Returns:
        404 Not Found
    """
    return Response(status_code=405)


async def handle_post_request(request: ApiRequest, assistant_id: str) -> Response:
    """Handle HTTP POST requests containing JSON-RPC messages.

    Args:
        request: The incoming HTTP request
        assistant_id: The assistant ID from the URL path

    Returns:
        JSON-RPC response
    """
    body = await request.body()

    try:
        message = orjson.loads(body)
    except orjson.JSONDecodeError:
        return create_error_response("Invalid JSON payload", 400)

    if not is_valid_accept_header(request):
        return create_error_response("Accept header must include application/json", 400)

    if not isinstance(message, dict):
        return create_error_response("Invalid message format", 400)

    if message.get("jsonrpc") != "2.0":
        return create_error_response(
            "Invalid JSON-RPC message. Missing or invalid jsonrpc version", 400
        )

    # Route based on message type
    id_ = message.get("id")
    method = message.get("method")

    if id_ is not None and method:
        # JSON-RPC request
        return await handle_jsonrpc_request(
            request, cast(JsonRpcRequest, message), assistant_id
        )
    elif id_ is not None:
        # JSON-RPC response (not expected in A2A server context)
        return handle_jsonrpc_response()
    elif method:
        # JSON-RPC notification
        return handle_jsonrpc_notification(cast(JsonRpcNotification, message))
    else:
        return create_error_response(
            "Invalid message format. Message must be a JSON-RPC request, "
            "response, or notification",
            400,
        )


def is_valid_accept_header(request: ApiRequest) -> bool:
    """Check if Accept header contains supported content types.

    Args:
        request: The incoming request

    Returns:
        True if header contains application/json
    """
    accept_header = request.headers.get("Accept", "")
    return "application/json" in accept_header


def create_error_response(message: str, status_code: int) -> Response:
    """Create a JSON error response.

    Args:
        message: Error message
        status_code: HTTP status code

    Returns:
        JSON error response
    """
    return Response(
        content=orjson.dumps({"error": message}),
        status_code=status_code,
        media_type="application/json",
    )


# ============================================================================
# JSON-RPC Message Handlers
# ============================================================================


async def handle_jsonrpc_request(
    request: ApiRequest, message: JsonRpcRequest, assistant_id: str
) -> Response:
    """Handle JSON-RPC requests with A2A methods.

    Args:
        request: The HTTP request
        message: Parsed JSON-RPC request
        assistant_id: The assistant ID from the URL path

    Returns:
        JSON-RPC response
    """
    method = message["method"]
    params = message.get("params", {})

    # Route to appropriate A2A method handler
    if method == "message/send":
        result_or_error = await handle_message_send(request, params, assistant_id)
    elif method == "tasks/get":
        result_or_error = await handle_tasks_get(request, params)
    elif method == "tasks/cancel":
        result_or_error = await handle_tasks_cancel(request, params)
    else:
        result_or_error = {
            "error": {
                "code": ERROR_CODE_METHOD_NOT_FOUND,
                "message": f"Method not found: {method}",
            }
        }

    response_keys = set(result_or_error.keys())
    if not (response_keys == {"result"} or response_keys == {"error"}):
        raise AssertionError(
            "Internal server error. Invalid response format in A2A implementation"
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": message["id"],
            **result_or_error,
        }
    )


def handle_jsonrpc_response() -> Response:
    """Handle JSON-RPC responses (not expected in server context).

    Args:
        message: Parsed JSON-RPC response

    Returns:
        202 Accepted acknowledgement
    """
    return Response(status_code=202)


def handle_jsonrpc_notification(message: JsonRpcNotification) -> Response:
    """Handle JSON-RPC notifications.

    Args:
        message: Parsed JSON-RPC notification

    Returns:
        202 Accepted acknowledgement
    """
    return Response(status_code=202)


# ============================================================================
# A2A Method Implementations
# ============================================================================


async def handle_message_send(
    request: ApiRequest, params: dict[str, Any], assistant_id: str
) -> dict[str, Any]:
    """Handle message/send requests to create or continue tasks.

    This method:
    1. Accepts A2A Messages containing text/file/data parts
    2. Maps to LangGraph assistant execution
    3. Returns Task objects with status and results

    Args:
        request: HTTP request for auth/headers
        params: A2A MessageSendParams
        assistant_id: The target assistant ID from the URL

    Returns:
        {"result": Task} or {"error": JsonRpcErrorObject}
    """
    client = _client()

    try:
        message = params.get("message")
        if not message:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Missing 'message' in params",
                }
            }

        parts = message.get("parts", [])
        if not parts:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Message must contain at least one part",
                }
            }

        try:
            assistant = await _get_assistant(client, assistant_id, request.headers)
            await _validate_supports_messages(client, assistant, request.headers, parts)
        except ValueError as e:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": str(e),
                }
            }

        # Process A2A message parts into LangChain messages format
        try:
            message_role = message.get(
                "role", "user"
            )  # Default to "user" if role not specified
            input_content = _process_a2a_message_parts(parts, message_role)
        except ValueError as e:
            return {
                "error": {
                    "code": ERROR_CODE_CONTENT_TYPE_NOT_SUPPORTED,
                    "message": str(e),
                }
            }

        context_id = message.get("contextId")
        thread_id = context_id if context_id else None

        try:
            # Creating + joining separately so we can get the run id
            run = await client.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=input_content,
                headers=request.headers,
            )
        except Exception as e:
            error_response = _map_runs_create_error_to_rpc(e, assistant_id, thread_id)
            if error_response.get("error", {}).get("code") == ERROR_CODE_INTERNAL_ERROR:
                raise
            return error_response

        result = await client.runs.join(
            thread_id=run["thread_id"],
            run_id=run["run_id"],
            headers=request.headers,
        )

        task_id = run["run_id"]
        context_id = thread_id or _generate_context_id()

        return _create_task_response(
            task_id=task_id,
            context_id=context_id,
            message=message,
            result=result,
            assistant_id=assistant_id,
        )

    except Exception as e:
        logger.exception(f"Error in message/send for assistant {assistant_id}")
        return {
            "error": {
                "code": ERROR_CODE_INTERNAL_ERROR,
                "message": f"Internal server error: {str(e)}",
            }
        }


async def handle_tasks_get(
    request: ApiRequest, params: dict[str, Any]
) -> dict[str, Any]:
    """Handle tasks/get requests to retrieve task status.

    This method:
    1. Accepts task ID from params
    2. Maps to LangGraph run/thread status
    3. Returns current Task state and results

    Args:
        request: HTTP request for auth/headers
        params: A2A TaskQueryParams containing task ID

    Returns:
        {"result": Task} or {"error": JsonRpcErrorObject}
    """
    client = _client()

    try:
        task_id = params.get("id")
        context_id = params.get("contextId")

        if not task_id:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Missing required parameter: id (task_id)",
                }
            }

        if not context_id:
            return {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Missing required parameter: contextId (thread_id)",
                }
            }

        try:
            run_info = await client.runs.get(
                thread_id=context_id,
                run_id=task_id,
                headers=request.headers,
            )
        except Exception as e:
            error_response = _map_runs_get_error_to_rpc(e, task_id, context_id)
            if error_response.get("error", {}).get("code") == ERROR_CODE_INTERNAL_ERROR:
                # For unmapped errors, re-raise to be caught by outer exception handler
                raise
            return error_response

        assistant_id = run_info.get("assistant_id")
        if assistant_id:
            try:
                # Verify that the assistant exists
                await _get_assistant(client, assistant_id, request.headers)
            except ValueError as e:
                return {
                    "error": {
                        "code": ERROR_CODE_INVALID_PARAMS,
                        "message": str(e),
                    }
                }

        lg_status = run_info.get("status", "unknown")

        if lg_status == "pending":
            a2a_state = "submitted"
        elif lg_status == "running":
            a2a_state = "working"
        elif lg_status == "success":
            a2a_state = "completed"
        elif lg_status in ["error", "timeout", "interrupted"]:
            a2a_state = "failed"
        else:
            a2a_state = "submitted"

        # Build the A2A Task response
        task_response = {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": a2a_state,
            },
        }

        # Add result message if completed
        if a2a_state == "completed":
            task_response["status"]["message"] = {
                "role": "agent",
                "parts": [{"kind": "text", "text": "Task completed successfully"}],
                "messageId": _generate_task_id(),
                "taskId": task_id,
            }
        elif a2a_state == "failed":
            task_response["status"]["message"] = {
                "role": "agent",
                "parts": [
                    {"kind": "text", "text": f"Task failed with status: {lg_status}"}
                ],
                "messageId": _generate_task_id(),
                "taskId": task_id,
            }

        return {"result": task_response}

    except Exception as e:
        await logger.aerror(
            f"Error in tasks/get for task {params.get('id')}: {str(e)}", exc_info=True
        )
        return {
            "error": {
                "code": ERROR_CODE_INTERNAL_ERROR,
                "message": f"Internal server error: {str(e)}",
            }
        }


async def handle_tasks_cancel(
    request: ApiRequest, params: dict[str, Any]
) -> dict[str, Any]:
    """Handle tasks/cancel requests to cancel running tasks.

    This method:
    1. Accepts task ID from params
    2. Maps to LangGraph run cancellation
    3. Returns updated Task with canceled state

    Args:
        request: HTTP request for auth/headers
        params: A2A TaskIdParams containing task ID

    Returns:
        {"result": Task} or {"error": JsonRpcErrorObject}
    """
    # TODO: Implement tasks/cancel
    # - Extract task_id from params
    # - Map task_id to run_id
    # - Cancel run via client if possible
    # - Return updated Task with canceled status

    return {
        "error": {
            "code": ERROR_CODE_UNSUPPORTED_OPERATION,
            "message": "Task cancellation is not currently supported",
        }
    }


# ============================================================================
# Agent Card Generation
# ============================================================================


async def generate_agent_card(request: ApiRequest, assistant_id: str) -> dict[str, Any]:
    """Generate A2A Agent Card for a specific assistant.

    Each LangGraph assistant becomes its own A2A agent with a dedicated
    agent card describing its individual capabilities and skills.

    Args:
        request: HTTP request for auth/headers
        assistant_id: The specific assistant ID to generate card for

    Returns:
        A2A AgentCard dictionary for the specific assistant
    """
    client = _client()

    assistant = await _get_assistant(client, assistant_id, request.headers)
    schemas = await client.assistants.get_schemas(assistant_id, headers=request.headers)

    # Extract schema information for metadata
    input_schema = schemas.get("input_schema", {})
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    assistant_name = assistant["name"]
    assistant_description = assistant.get("description", f"{assistant_name} assistant")

    # For now, each assistant has one main skill - itself
    skills = [
        {
            "id": f"{assistant_id}-main",
            "name": f"{assistant_name} Capabilities",
            "description": assistant_description,
            "tags": ["assistant", "langgraph"],
            "examples": [],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json", "text/plain"],
            "metadata": {
                "inputSchema": {
                    "required": required,
                    "properties": sorted(properties.keys()),
                    "supportsA2A": "messages" in properties,
                }
            },
        }
    ]

    if USER_API_URL:
        base_url = USER_API_URL.rstrip("/")
    else:
        # Fallback to constructing from request
        scheme = request.url.scheme
        host = request.url.hostname or "localhost"
        port = request.url.port
        if port and (
            (scheme == "http" and port != 80) or (scheme == "https" and port != 443)
        ):
            base_url = f"{scheme}://{host}:{port}"
        else:
            base_url = f"{scheme}://{host}"

    return {
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "name": assistant_name,
        "description": assistant_description,
        "url": f"{base_url}/a2a/{assistant_id}",
        "preferredTransport": "JSONRPC",
        "capabilities": {
            "streaming": False,  # Not implemented yet
            "pushNotifications": False,  # Not implemented yet
            "stateTransitionHistory": False,
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": skills,
        "version": _get_version(),
    }


async def handle_agent_card_endpoint(request: ApiRequest) -> Response:
    """Serve Agent Card for a specific assistant.

    Expected URL: /.well-known/agent-card.json?assistant_id=uuid

    Args:
        request: HTTP request

    Returns:
        JSON response with Agent Card for the specific assistant
    """
    try:
        # Get assistant_id from query parameters
        assistant_id = request.query_params.get("assistant_id")

        if not assistant_id:
            error_response = {
                "error": {
                    "code": ERROR_CODE_INVALID_PARAMS,
                    "message": "Missing required query parameter: assistant_id",
                }
            }
            return Response(
                content=orjson.dumps(error_response),
                status_code=400,
                media_type="application/json",
            )

        agent_card = await generate_agent_card(request, assistant_id)
        return JSONResponse(agent_card)

    except ValueError as e:
        # A2A validation error or assistant not found
        error_response = {
            "error": {
                "code": ERROR_CODE_INVALID_PARAMS,
                "message": str(e),
            }
        }
        return Response(
            content=orjson.dumps(error_response),
            status_code=400,
            media_type="application/json",
        )
    except Exception as e:
        logger.exception("Failed to generate agent card")
        error_response = {
            "error": {
                "code": ERROR_CODE_INTERNAL_ERROR,
                "message": f"Internal server error: {str(e)}",
            }
        }
        return Response(
            content=orjson.dumps(error_response),
            status_code=500,
            media_type="application/json",
        )


# ============================================================================
# Route Definitions
# ============================================================================


async def handle_a2a_assistant_endpoint(request: ApiRequest) -> Response:
    """A2A endpoint handler for specific assistant.

    Expected URL: /a2a/{assistant_id}

    Args:
        request: The incoming HTTP request

    Returns:
        JSON-RPC response or appropriate HTTP error response
    """
    # Extract assistant_id from URL path params
    assistant_id = request.path_params.get("assistant_id")
    if not assistant_id:
        return create_error_response("Missing assistant ID in URL", 400)

    if request.method == "POST":
        return await handle_post_request(request, assistant_id)
    elif request.method == "GET":
        return handle_get_request()
    elif request.method == "DELETE":
        return handle_delete_request()
    else:
        return Response(status_code=405)  # Method Not Allowed


a2a_routes = [
    # Per-assistant A2A endpoints: /a2a/{assistant_id}
    ApiRoute(
        "/a2a/{assistant_id}",
        handle_a2a_assistant_endpoint,
        methods=["GET", "POST", "DELETE"],
    ),
    ApiRoute(
        "/.well-known/agent-card.json", handle_agent_card_endpoint, methods=["GET"]
    ),
]
