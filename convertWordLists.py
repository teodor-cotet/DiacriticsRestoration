if __name__ == "__main__":
    with open("wordlists.txt", "rt") as f:
        lists = [line.strip() for line in f.readlines()]
    with open("wordlists-nl.csv", "rt") as f, open("wordlists-nl-new.csv", "wt") as out:
        f.readline()
        last = ""
        out.write("sep=;\n")
        out.write("word;" + ";".join(lists) + "\n")
        current = {}
        for line in f.readlines():
            word, valence, score = line.strip().split(";")
            if last != word:
                if last != "":
                    out.write(last + ";" + ";".join([str(current[wl]) if wl in current else "0" for wl in lists]) + "\n")
                current = {valence: score}
                last = word
            else:
                current[valence] = score
        out.write(last + ";" + ";".join([str(current[wl]) if wl in current else "0" for wl in lists]) + "\n")
                