import difflib
import re

def clean_code(code_str):
    code_str = re.sub(r'#.*', '', code_str)
    code_str = re.sub(r"'''[\s\S]*?'''", "", code_str)
    code_str = re.sub(r'"""[\s\S]*?"""', "", code_str)

    lines = [line.strip() for line in code_str.splitlines() if line.strip()]
    
    return '\n'.join(lines)

def calculate_similarity(file1_path, file2_path):

    try:

        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            code1 = f1.read()
            code2 = f2.read()


        clean_code1 = clean_code(code1)
        clean_code2 = clean_code(code2)


        matcher = difflib.SequenceMatcher(None, clean_code1, clean_code2)
        similarity_ratio = matcher.ratio()

        return similarity_ratio

    except FileNotFoundError as e:
        return f"Error - File Not Found: {e.filename}"
    except Exception as e:
        return f"Error - Unknown Error: {e}"

if __name__ == "__main__":
    file_a = "code1.py"
    file_b = "code4.py"

    ratio = calculate_similarity(file_a, file_b)
    
    if isinstance(ratio, float):
        print(f"-> Code duplication rate is {ratio * 100:.2f}%")
    else:
        print(ratio)