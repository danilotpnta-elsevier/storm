def extract_url(text):
    # This pattern matches the format [number]: text (url)
    pattern = r"\[\d+\]: .* \((https?://[^\)]+)\)"

    # Find the first match of the pattern in the text
    match = re.search(pattern, text)

    # Return the URL if a match is found
    if match:
        return match.group(1)
    else:
        return None