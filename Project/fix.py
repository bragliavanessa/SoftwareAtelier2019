import os
import sys
import shutil


def find_files(start_dir):
    ext_paths = []
    for root, dirs, files in os.walk(start_dir):
        if root == start_dir:
            continue
        ext_paths = ext_paths + \
            list(map(lambda x: os.path.join(root, x), files))
    return ext_paths


def alter_path(path, new_dir):
    newpath = path.split('\\')[1:]
    newpath = '_'.join(newpath)
    print(newpath)
    return os.path.join(new_dir, newpath)


def main():
    for i in find_files(sys.argv[1]):
        # print(i, alter_path(i, sys.argv[1]))
        shutil.copy(i, alter_path(i, sys.argv[1]))


if __name__ == '__main__':
    main()
