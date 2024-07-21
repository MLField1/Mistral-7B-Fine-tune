import re
import json

def process_file(input_file, output_file):
    # Read the input file with UTF-8 encoding
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Remove IEEE reference numbers
    text = re.sub(r'\[\d+\]', '', text)

    # Remove newline characters
    text = text.replace('\n', ' ')

    # Improved sentence splitting regex
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'

    # Split the text into sentences
    sentences = re.split(sentence_pattern, text)

    # Write sentences to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Ignore empty sentences
                # Remove periods inside parentheses at the end of the sentence
                sentence = re.sub(r'\([^)]*\.\)$', lambda m: m.group().replace('.', ''), sentence)
                json_line = json.dumps({"note": sentence}, ensure_ascii=False)
                f.write(json_line + '\n')

# Example usage
input_file = 'input.txt'
output_file = 'output.jsonl'
process_file(input_file, output_file)
