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
def search_documents(query: str, filenames: str = None) -> str:
    """
    Useful for searching the uploaded policy manuals, guidelines, employee handbooks,
    and other corporate documentation to retrieve factual information about the company.
    If specific documents are the target of the search, pass their filenames as a comma-separated string (e.g. 'doc1.pdf,doc2.pdf').

    Args:
        query: The semantic search query targeting policy details.
        filenames: Comma-separated list of filenames to restrict the search to. ONLY pass this if you are explicitly instructed to search specific filenames or if the filenames are present in the selection list. DO NOT guess, assume, or hallucinate filenames (such as 'policy.pdf') if not provided. Leave empty to search all documents.

    Returns:
        A consolidated text string of the retrieved document chunks with appended source and page citations,
        or 'no relevant informations found for it.' if no matches are found.
    """
    try:
        parsed_files = [f.strip() for f in filenames.split(",")] if filenames else None
        docs = retrieve_documents(query, top_k=4, filenames=parsed_files)
        if not docs:
            return "no relevant informations found for it."

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
def web_search(query: str) -> str:
    """
    Useful for searching the live internet using DuckDuckGo to answer questions about
    current events, real-time facts, news, or general knowledge.
    CRITICAL: ONLY call this tool if the user has explicitly confirmed or requested a web search in response to a prompt.
    Do NOT call this tool automatically.

    Args:
        query: The search query targeting live web facts.

    Returns:
        A formatted string of relevant snippets and links from the web search.
    """
    # Method 1: HTML scraping (bypasses TLS/API blocks)
    try:
        import urllib.parse
        import requests
        from bs4 import BeautifulSoup

        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for result in soup.find_all("div", class_="result__body")[:5]:
                title_tag = result.find("a", class_="result__a")
                snippet_tag = result.find("a", class_="result__snippet")
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    href = title_tag.get("href", "")
                    
                    # Clean redirect URLs (extract direct link from uddg query parameter)
                    if "uddg=" in href:
                        parsed_href = urllib.parse.urlparse(href)
                        params = urllib.parse.parse_qs(parsed_href.query)
                        if "uddg" in params:
                            href = params["uddg"][0]
                    elif href.startswith("//"):
                        href = "https:" + href

                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    results.append({
                        "title": title,
                        "href": href,
                        "body": snippet
                    })
            if results:
                formatted = []
                for r in results:
                    formatted.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
                return "\n---\n".join(formatted)
    except Exception:
        pass

    # Method 2: Fallback to duckduckgo_search library
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            if not results:
                return f"No live search results found for: '{query}'"
            formatted = []
            for r in results:
                title = r.get("title", "No Title")
                link = r.get("href", "")
                body = r.get("body", "")
                formatted.append(f"Title: {title}\nURL: {link}\nSnippet: {body}\n")
            return "\n---\n".join(formatted)
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def summarize_document_topic(topic: str, filenames: str = None) -> str:
    """
    Useful for retrieving a broad overview, summary, or synthesis of a specific topic,
    theme, or concept across uploaded documents or specific documents.
    This tool retrieves a larger context window (up to 8 chunks) to extract general themes.

    Args:
        topic: The topic, theme, or concept to summarize from the documents.
        filenames: Comma-separated list of filenames to restrict the summary to. ONLY pass this if you are explicitly instructed to summarize specific filenames or if the filenames are present in the selection list. DO NOT guess, assume, or hallucinate filenames (such as 'doc1.pdf') if not provided. Leave empty to search all documents.

    Returns:
        A consolidated summary string of retrieved chunks with their source and page metadata,
        or 'no relevant informations found for it.' if no matches are found.
    """
    try:
        parsed_files = [f.strip() for f in filenames.split(",")] if filenames else None
        docs = retrieve_documents(topic, top_k=8, filenames=parsed_files)
        if not docs:
            return "no relevant informations found for it."

        results = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown file")
            page = doc.metadata.get("page", 1)
            results.append(f"Content: {doc.page_content} ... (Source: {source}, Page: {page})\n\n")

        return "".join(results)
    except Exception as e:
        return f"Error summarizing document topic: {str(e)}"
