import ast
import operator
from datetime import datetime
from langchain_core.tools import tool
from app.services.vector_store import retrieve_documents

# Safe AST Calculator Implementation
# Prevents Remote Code Execution (RCE) by restricting evaluation to basic math operations
_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    elif isinstance(node, ast.Constant):  # Python >= 3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        op_type = type(node.op)
        if op_type in _operators:
            return _operators[op_type](left, right)
        raise TypeError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast_node(node.operand)
        op_type = type(node.op)
        if op_type in _operators:
            return _operators[op_type](operand)
        raise TypeError(f"Unsupported unary operator: {op_type}")
    raise TypeError(f"Unsupported expression component: {type(node)}")

def _safe_eval(expr_str: str) -> float:
    """
    Safely parses and evaluates mathematical expressions using AST.
    """
    clean_expr = "".join(expr_str.split())
    tree = ast.parse(clean_expr, mode="eval")
    result = _eval_ast_node(tree)
    return result

@tool
def search_documents(query: str) -> str:
    """
    Useful for searching the uploaded policy manuals, guidelines, employee handbooks,
    and other corporate documentation to retrieve factual information about the company.

    Args:
        query: The semantic search query targeting policy details.

    Returns:
        A consolidated text string of the retrieved document chunks with appended source and page citations.
    """
    try:
        docs = retrieve_documents(query, top_k=4)
        if not docs:
            return "No matching corporate documents or policies were found."

        results = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown file")
            page = doc.metadata.get("page", 1)
            results.append(f"Content: {doc.page_content} ... (Source: {source}, Page: {page})\n\n")

        return "".join(results)
    except Exception as e:
        return f"Error executing document search: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """
    Useful for evaluating mathematical expressions.
    Input must be a valid mathematical string containing only numbers and operators (+, -, *, /).
    Do not pass conversational text, variable names, or letters (except exponent notation).

    Args:
        expression: The mathematical expression string to calculate (e.g., '50000 * 0.15').

    Returns:
        The calculated numerical result as a string, or an error message.
    """
    try:
        expr_clean = expression.replace("x", "*").replace("^", "**")
        result = _safe_eval(expr_clean)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"

@tool
def get_current_date_time() -> str:
    """
    Useful for retrieving the current local system date and time.
    Use this tool whenever the user's query refers to relative times such as 'today',
    'yesterday', 'now', or requires computing dates relative to the current timestamp.

    Returns:
        The current date and time formatted as a string.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
