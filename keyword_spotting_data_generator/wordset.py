import json
import requests

# https://www.wordsapi.com/

API_URL = "https://wordsapiv1.p.rapidapi.com/words/"
API_KEY = "WORDS_API_KEY"

def get_relevant_words(search_term, depth=1):
	relevant_arr = [search_term]
	to_be_searched = [search_term]

	while depth > 0 and len(to_be_searched) > 0:
		next_to_be_searched = []
		for word in to_be_searched:
			response = requests.get(API_URL+word, headers={"X-RapidAPI-Key": API_KEY}).json()
			if "results" not in response:
				continue

			for result in response["results"]:
				if "similarTo" in result:
					for similar in result["similarTo"]:
						if similar not in relevant_arr and similar not in to_be_searched:
							next_to_be_searched.append(similar)

				if "derivation" in result:
					for derivation in result["derivation"]:
						if derivation not in relevant_arr and derivation not in to_be_searched:
							next_to_be_searched.append(derivation)

				if "synonyms" in result:
					for synonym in result["synonyms"]:
						if synonym not in relevant_arr and synonym not in to_be_searched:
							next_to_be_searched.append(synonym)

			relevant_arr += next_to_be_searched			

		to_be_searched = next_to_be_searched
		depth -= 1

	return relevant_arr