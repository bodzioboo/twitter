#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:07:44 2020

@author: piotr
"""

import csv
import itertools
import datetime
import os
import argparse
import re

def drop_date(path_source, path_target, date):
    
    
        
    with open(path_source,"r") as f1:
        reader = csv.DictReader(f1)
        
        
        for _ in itertools.count():
            
            
            check = list(itertools.islice(reader, 100000))                 
            if not check:
                break
                
                
            fmt = "%a %b %d %H:%M:%S +0000 %Y"
            condition = datetime.datetime.strptime(date, "%Y/%m/%d")
                    #date is not text:
            redate = "[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}"
            #pdb.set_trace()
            check = [elem for elem in check if re.match(redate, str(elem["created_at"]))]
            check = [elem for elem in check if datetime.datetime.strptime(elem["created_at"],fmt) < condition]
            
            length_out = len(check)
            
        
            if check:
            
                write_mode = "a" if os.path.isfile(path_target) else "w"
                with open(path_target,write_mode) as f2:
                    fieldnames = list(check[0].keys())
                    writer = csv.DictWriter(f2, fieldnames = fieldnames)
                    if write_mode == "w":
                        writer.writeheader()
                    writer.writerows(check)
                print(f"Wrote {length_out} to file {path_target}")
            
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("path_source")
        parser.add_argument("path_target")
        parser.add_argument("date", help = "YYYY/mm/dd format")
        args = parser.parse_args()
        drop_date(args.path_source, args.path_target, args.date)
    except Exception as e:
        print(e)
    

    
   
        