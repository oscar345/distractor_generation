# small test to check if function and results are correct

from datasets import load_from_disk
from pprint import pprint


dataset = load_from_disk("./project/predictions/mistral")

pprint([row for row in dataset.select(range(30))])
