import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.table import Table
# List of common stop words to exclude
STOP_WORDS = set(['the', 'and', 'is', 'a', 'an', 'in', 'to', 'of', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into',
  'over', 'after', 'or','this','are', 'where', 'that','each', 'can','these','be','which','through'])

def read_file(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode the file {file_path} with any of the attempted encodings.")


def get_words(text):
    return set(word.lower() for word in re.findall(r'\w+', text) if word.lower() not in STOP_WORDS)

def get_Allwords(text):
    return set(word.lower() for word in re.findall(r'\w+', text) if word.lower())


def count_words(text):
    words = re.findall(r'\w+', text.lower())
    return len(words)


def count_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def average_words_per_sentence(text):
    words = count_words(text)
    sentences = count_sentences(text)
    return words / sentences if sentences > 0 else 0


def average_sentences_per_paragraph(text):
    paragraphs = text.split('\n\n')
    sentence_counts = [count_sentences(p) for p in paragraphs if p.strip()]
    return sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0


def most_common_words(text, n=10):
    words = re.findall(r'\w+', text.lower())
    return Counter(word for word in words if word not in STOP_WORDS).most_common(n)

def most_common_Allwords(text, n=10):
    words = re.findall(r'\w+', text.lower())
    return Counter(word for word in words).most_common(n)

def analyze_paragraphs(text):
    paragraphs = text.split('\n\n')
    return [count_words(p) for p in paragraphs[:5]]  # Limit to first 5 paragraphs


def compare_vocabulary(text1, text2):
    words1 = get_words(text1)
    words2 = get_words(text2)
    unique_to_1 = words1 - words2
    unique_to_2 = words2 - words1
    return unique_to_1, unique_to_2

def compare_Allvocabulary(text1, text2):
    words1 = get_Allwords(text1)
    words2 = get_Allwords(text2)
    uniqueAll_to_1 = words1 - words2
    uniqueAll_to_2 = words2 - words1
    return uniqueAll_to_1, uniqueAll_to_2

def add_value_labels(ax, spacing=5):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        label = f"{y_value:.1f}"
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                    textcoords="offset points", ha='center', va='bottom')


def plot_unique_words_table(unique_to_1, unique_to_2):
    # Sort the unique words alphabetically
    sorted_words_1 = sorted(unique_to_1)
    sorted_words_2 = sorted(unique_to_2)

    # Determine the number of rows needed
    max_rows = max(len(sorted_words_1), len(sorted_words_2))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, max_rows * 0.3))  # Adjust figure size based on number of words
    ax.axis('off')

    # Create the table
    table = Table(ax, bbox=[0, 0, 1, 1])

    # Add header
    table.add_cell(0, 0, 0.5, 0.05, text="Mistral 7B Instruct Fine-Tune", loc='center', facecolor='lightblue')
    table.add_cell(0, 1, 0.5, 0.05, text="Mistral 7B Instruct", loc='center', facecolor='lightblue')

    # Add data
    for i in range(max_rows):
        word_1 = sorted_words_1[i] if i < len(sorted_words_1) else ""
        word_2 = sorted_words_2[i] if i < len(sorted_words_2) else ""
        table.add_cell(i+1, 0, 0.5, 0.02, text=word_1)
        table.add_cell(i+1, 1, 0.5, 0.02, text=word_2)

    # Add the table to the plot
    ax.add_table(table)

    # Add a title
    plt.title("Complete List of Unique Words", pad=20)

    # Save the figure
    plt.savefig('unique_words_table.png', bbox_inches='tight', dpi=300)
    plt.close()


def analyze_paragraphs(text):
    # Split the text into lines and consider each line a paragraph
    paragraphs = text.split('\n')
    # Count words in each paragraph (line)
    return [count_words(p) for p in paragraphs if p.strip()]

def analyze_file(file_path):
    text = read_file(file_path)
    avg_words = average_words_per_sentence(text)
    avg_sentences = average_sentences_per_paragraph(text)
    common_words = most_common_words(text)
    common_Allwords = most_common_Allwords(text)
    paragraph_words = analyze_paragraphs(text)
    return text, avg_words, avg_sentences, common_words,common_Allwords, paragraph_words
def output_unique_words_to_file(unique_to_1, unique_to_2, output_filename='unique_words.txt'):
    # Sort the unique words alphabetically
    sorted_words_1 = sorted(unique_to_1)
    sorted_words_2 = sorted(unique_to_2)

    # Determine the maximum number of words
    max_words = max(len(sorted_words_1), len(sorted_words_2))

    # Pad the shorter list with empty strings
    sorted_words_1 += [''] * (max_words - len(sorted_words_1))
    sorted_words_2 += [''] * (max_words - len(sorted_words_2))

    # Write to file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("Mistral 7B Instruct Fine-Tune | Mistral 7B Instruct\n")
        f.write("-" * 60 + "\n")
        for word1, word2 in zip(sorted_words_1, sorted_words_2):
            f.write(f"{word1:<30} | {word2}\n")
def plot_results(file1_results, file2_results):
    _, avg_words1, avg_sentences1, common_words1, common_Allwords1,paragraph_words1 = file1_results
    _, avg_words2, avg_sentences2, common_words2, common_Allwords2,paragraph_words2 = file2_results

    # Plot average words per sentence and average sentences per paragraph
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(12, 16))

    metrics = ['Avg Words per Sentence', 'Avg Sentences per Paragraph']
    values1 = [avg_words1, avg_sentences1]
    values2 = [avg_words2, avg_sentences2]

    x = range(len(metrics))
    width = 0.35

    ax1.bar([i - width/2 for i in x], values1, width, label='Mistral 7B Instruct Fine-Tune')
    ax1.bar([i + width/2 for i in x], values2, width, label='Mistral 7B Instruct')

    y_max = max(max(values1), max(values2))
    ax1.set_ylim(0, y_max * 1.2)
    ax1.set_ylabel('Value')
    ax1.set_title('Average Words per Sentence and Sentences per Paragraph')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()

    add_value_labels(ax1)

    # Plot most common words
    words1, counts1 = zip(*common_words1)
    words2, counts2 = zip(*common_words2)

    x1 = range(len(words1))
    x2 = range(len(words2))

    y_max2 = max(max(counts1), max(counts2))
    ax2.set_ylim(0, y_max2 * 1.2)
    ax2.bar(x1, counts1, align='center', width=0.5,label='Mistral 7B Instruct Fine-Tune')
    ax2.bar(x2, counts2, align='edge' ,width=0.5, label='Mistral 7B Instruct')

    ax2.set_ylabel('Frequency')
    ax2.set_title('Most Common Words')
    ax2.set_xticks(x1)
    ax2.set_xticklabels(words1, rotation=45, ha='right')
    ax2.legend()

    add_value_labels(ax2)

    # Plot most common words
    wordsAll1, countsAll1 = zip(*common_Allwords1)
    wordsAll2, countsAll2 = zip(*common_Allwords2)

    x11 = range(len(wordsAll1))
    x21 = range(len(wordsAll2))

    y_max3 = max(max(countsAll1), max(countsAll2))
    ax3.set_ylim(0, y_max3 * 1.2)

    #Plot of all most common words
    ax3.bar(x11, countsAll1, align='center', width=0.5, label='Mistral 7B Instruct Fine-Tune')
    ax3.bar(x21, countsAll2, align='edge', width=0.5, label='Mistral 7B Instruct')

    ax3.set_ylabel('Frequency')
    ax3.set_title('Most Common Words')
    ax3.set_xticks(x11)
    ax3.set_xticklabels(wordsAll1, rotation=45, ha='right')
    ax3.legend()

    add_value_labels(ax3)

    # Calculate the number of paragraphs to plot (use the shorter list)
    num_paragraphs = min(len(paragraph_words1), len(paragraph_words2))

    # Prepare data for plotting
    x = range(1, num_paragraphs + 1)
    y1 = paragraph_words1[:num_paragraphs]
    y2 = paragraph_words2[:num_paragraphs]

    y_max4 = max(max(y1), max(y2))
    ax4.set_ylim(0, y_max4 * 1.2)

    # Plot the data
    ax4.bar(x, y1, align='center', width=0.5, label='Mistral 7B Instruct Fine-Tune')
    ax4.bar(x, y2,align='edge',width=0.5, label='Mistral 7B Instruct')

    ax4.set_xlabel('Paragraph Number')
    ax4.set_ylabel('Word Count')
    ax4.set_title('Word Count per Paragraph')
    ax4.legend()

    # Adjust x-axis ticks
    ax4.set_xticks(x)
    add_value_labels(ax4)

    plt.tight_layout()
    plt.savefig('text_analysis_results.png')
    plt.close()

# Main execution
file1_path = 'text_file_1.txt'
file2_path = 'text_file_2.txt'

try:
    file1_results = analyze_file(file1_path)
    file2_results = analyze_file(file2_path)

    plot_results(file1_results, file2_results)

    text1, avg_words1, avg_sentences1, common_words1, common_Allwords1,paragraph_words1 = file1_results
    text2, avg_words2, avg_sentences2, common_words2, common_Allwords2,paragraph_words2 = file2_results

    # Compare vocabularies
    unique_to_1, unique_to_2 = compare_vocabulary(text1, text2)

    # Create table of all unique words
    plot_unique_words_table(unique_to_1, unique_to_2)

    output_unique_words_to_file(unique_to_1, unique_to_2)
    print("\nUnique words have been written to 'unique_words.txt'")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
