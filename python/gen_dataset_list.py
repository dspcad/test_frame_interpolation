import os, sys
import argparse
import csv

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate the dataset list')

parser.add_argument('--csv_file', type=str, 
                    help='A required bin file for frame interpolation')

parser.add_argument('--output', type=str, 
                    help='A required bin file for frame interpolation')



args = parser.parse_args()


if(not os.path.isfile(args.csv_file)):
    print(f"{args.csv_file} cannot be found ... make sure csv file exists")
    sys.exit(1)



with open(args.csv_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    lines = list(spamreader)



root_dir = "/NFS_DATA/dataset/vendor/MOBILE"
#with open("wzry_data_list.txt","w") as f:
with open(args.output,"w") as f:
    for line in lines[1:]:
        if line[-1]=="True":
            path = line[0].split("/")
            new_path=[root_dir]
            new_path.extend(path[4:-1])
            game_date = path[-1].removesuffix('.ofpc')
            new_path.append(game_date)
            new_path.append("view")
            s = "/"
            s = s.join(new_path)
            f.write(f"{s}\n")



