#!/usr/bin/python3
import argparse

ps = argparse.ArgumentParser(description='Automatically changes pipeline config variables for mass pipeline modification')
ps.add_argument("pipeline", type=str, help='Path to pipeline pipeline.config')
ps.add_argument("-o", "--outfile", type=str, help="Path to write new pipeline.config. If not defined, overwrite infile.")

ps.add_argument("-c", "--checkpoint", type=str, help="Path to checkpoint checkpoint.ckpt")
ps.add_argument("-l", "--labelmap", type=str, help="Path to labelmap labelmap.pbtxt")

ps.add_argument("-t", "--trainr", type=str, help="Path to training record training.record")
ps.add_argument("-e", "--evalr", type=str, help="Path to eval record eval.record")

args=ps.parse_args()

ts = None

try:
	pipeline = open(args.pipeline, "r").readlines()
except IOError as error:
	print("Could not load "+args.pipeline)
	exit()

for line in pipeline:
	line=line.strip("\n")
	if "label_map_path: " in line and args.labelmap:
		pipeline = [item.replace(line, line.split(": ")[0]+": \""+args.labelmap+"\"") for item in pipeline]
	if "fine_tune_checkpoint: " in line and args.checkpoint:
		pipeline = [item.replace(line, line.split(": ")[0]+": \""+args.checkpoint+"\"") for item in pipeline]
	if "train_input_reader" in line:
		ts=True
	if "eval_input_reader" in line:
		ts=False
	if "input_path: " in line and ts is True and args.trainr:
		pipeline = [item.replace(line, line.split(": ")[0]+": \""+args.trainr+"\"") for item in pipeline]
	if "input_path: " in line and ts is False and args.evalr:
		pipeline = [item.replace(line, line.split(": ")[0]+": \""+args.evalr+"\"") for item in pipeline]

if args.outfile:
	if args.outfile[-1] == "/":
		args.outfile+"pipeline.config"
	outpipe = open(args.outfile, "w")
else:
	outpipe = open(args.pipeline, "w")
outpipe.write("".join(pipeline))
