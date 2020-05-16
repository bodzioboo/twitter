#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:18:46 2020

@author: piotr
"""

import csv
import itertools
import argparse
import os

def csv_bind(file1, file2, check_headers = True):
    
    if check_headers:
        #make sure that the column headings are the same
        f1cols = os.popen(f"head -1 {file1}").read()
        f2cols = os.popen(f"head -1 {file2}").read()
        assert(f1cols == f2cols)
    
    with open(file2, "r") as f2:
        reader = csv.DictReader(f2)
        next(reader)
    
        for _ in itertools.count():
            data = list(itertools.islice(reader, 100000))
            
            if not data:
                break
            
            with open(file1, "a") as f1:
                writer = csv.DictWriter(f1, fieldnames = data[0].keys())
                writer.writerows(data)
            print("Appended new rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1")
    parser.add_argument("file2")
    args = parser.parse_args()
    csv_bind(args.file1, args.file2)