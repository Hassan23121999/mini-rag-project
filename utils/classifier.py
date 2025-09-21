def classify_query(query: str, retriever) -> str:
    """Classify query based on keywords or if retriever can find related info in the CV."""
    cv_keywords = ["experience", "work", "skills", "education", "certifications", "degree"]
    
    # Check if the query contains CV-specific terms
    if any(word in query.lower() for word in cv_keywords):
        return "document"
    
    # Use retriever to find relevant information
    docs = retriever.get_relevant_documents(query)
    if len(docs) > 0:  # If retriever finds any relevant docs
        return "document"
    
    return "general"
