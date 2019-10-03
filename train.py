import argparse
from model import Model
import pdb
import pickle

if __name__ == "__main__":
	parser = argparse.ArgumentParser('Multi model text classification trainer')
	parser.add_argument('-m', help='model name', type=str, required=True)
	parser.add_argument('-i', help='input file', type=str, required=True)
	parser.add_argument('-o', help='output file', type=str, required=True)
	parser.add_argument('-s', help='score mode', type=bool, default=False)
	args = parser.parse_args()
	print('Args:', args)

	model = Model.create(args.m)
	if args.s:
		model.score(args.i)
	else:
		model.train(args.i)
		with open(args.o, 'wb') as fout:
			pickle.dump(model, fout, pickle.HIGHEST_PROTOCOL)

	print('OK')
