from pathlib import Path


def make_webdataset(in_dir:str, out_dir):
    import tarfile
    # print(Path(in_dir).glob("*"))
    in_folders = [x for x in Path(in_dir).glob("*") if x.is_dir()]
    # print(Path('outTest/aa.txt').is_file())
    print(in_folders)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for folder in in_folders:
        filename = out_dir / f"{folder.stem}.tar"
        files_to_add = sorted(list(folder.rglob("*")))

        with tarfile.open(filename, "w") as tar:
            for f in files_to_add:
                tar.add(f)

make_webdataset('outTest','outTest')