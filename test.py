file = "history_cleaned.txt"

def read_line(line):
    for line in open(file, "r", encoding="utf-8"):
        print(line.strip())


