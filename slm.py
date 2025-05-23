# ─────────────────────────────────────────────────────────────────────────────
# Description  : Script de génération de texte basé sur un modèle n-gram
#               et une interface en ligne de commande (CLI).
#               Le script permet d'analyser un corpus de texte, de construire
#               un modèle n-gram, de sauvegarder le modèle et de générer du texte
#               à partir du modèle.

# Date         : 2023-10-12
# Last Update  : 2023-10-12
# Version      : v1.0
# Authors      : VALLADE Allan,
#                CASAGRANDE Michael,
#                OUALI Mohamed,
#                CHAMBRE Ryan,
#                FALANDRY Enzo
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import json
import random
import re
import unicodedata
import gzip


def normalize_text(text, remove_accents=True):
    text = text.lower()
    # Suppression des accents
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # Suppression des chiffres et ponctuations et remplacer les retour charriot par des espaces
    text = re.sub(r'[\r\n]+', ' ', text)  # Remplace les retours à la ligne par des espaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Supprime les caractères non alphabétiques
    return text

def read_corpus(corpus_dir):
    corpus = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), encoding='utf8') as f:
                corpus += f.read() + "\n"
    return corpus

def build_ngram_model(text, n=3):
    model = {}

    for i in range(len(text) - n):
        ngram = text[i:i+n]            # Séquence de n caractères
        next_char = text[i + n]        # Caractère suivant

        if ngram not in model:
            model[ngram] = {}
        if next_char not in model[ngram]:
            model[ngram][next_char] = 0
        model[ngram][next_char] += 1

    return model

def save_model(model, output_file, compress=False):
    if compress:
        with gzip.open(output_file + '.gz', 'wt', encoding='utf8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        return output_file + '.gz'
    else:
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
    return output_file
def load_model(model_file):
    if model_file.endswith('.gz'):
        with gzip.open(model_file, 'rt', encoding='utf8') as f:
            model = json.load(f)
    elif model_file.endswith('.json'):
        with open(model_file, 'r', encoding='utf8') as f:
            model = json.load(f)
    
    else:
        raise ValueError("Le fichier de modèle doit être au format .json ou .gz")
    return model

def generate_text(model, seed, length):
    text = seed
    for _ in range(length):
        if seed not in model:
            break
        next_chars = model[seed]
        total = sum(next_chars.values())
        rand = random.randint(1, total)
        for char, count in next_chars.items():
            rand -= count
            if rand <= 0:
                text += char
                seed = text[-len(seed):]  # Met à jour le n-gramme
                break
    return text

# Exemple d'utilisation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLM CLI")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["analyse", "generate", "both"],
        help="Mode to run the SLM CLI. Choose from 'analyse', 'generate', or 'both'.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input file path for the SLM CLI. This is required for 'analyse' and 'generate' modes.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="output.txt",
        help="Output file path for the SLM CLI. This is required for 'analyse' and 'generate' modes.",
    )
    parser.add_argument(
        "--output-generation",
        type=str,
        default="output.txt",
        help="Output file path for the SLM CLI. This is required for 'analyse' and 'generate' modes.",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order, default is 1.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model file path for the SLM CLI. This is required for 'generate' mode.",
    )
    parser.add_argument(
        "--keep-spaces",
        type=bool,
        default=True,
        help="Keep spaces in the input text.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="",
        help="Seed for the random number generator. This is used to ensure reproducibility of results.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Seed for the random number generator. This is used to ensure reproducibility of results.",
    )

    parser.add_argument(
        "--compress",
        type=bool,
        default=False,
        help="Whether to compress the model file.",
    )

    args = parser.parse_args()
    if args.mode == "analyse" :
        #verifie les paramatres
        if not args.input:
            print("Le paramètre --input est requis pour le mode 'analyse'.")
            exit(1)
        if not os.path.exists(args.input):
            print(f"Le fichier d'entrée '{args.input}' n'existe pas.")
            exit(1)
        if not args.output_model:
            print("Le paramètre --output-model est requis pour le mode 'analyse'.")
            exit(1)
        if not args.order:
            print("Le paramètre --order est requis pour le mode 'analyse'.")
            exit(1)
        
        raw_text = read_corpus(args.input)
        clean_text = normalize_text(raw_text)
        print(f"Texte normalisé : {clean_text[:100]}...")  # Affiche les 100 premiers caractères du texte normalisé
        ngram_model = build_ngram_model(clean_text, args.order)
        save_model(ngram_model, args.output_model, compress=args.compress)
    elif args.mode == "generate" :
        #verifie les paramatres
        if not args.model:
            print("Le paramètre --model est requis pour le mode 'generate'.")
            exit(1)
        if not args.output_generation:
            print("Le paramètre --output-generation est requis pour le mode 'generate'.")
            exit(1)
        if not os.path.exists(args.model):
            print(f"Le fichier de modèle '{args.model}' n'existe pas.")
            exit(1)
        if not args.seed:
            print("Le paramètre --seed est requis pour le mode 'generate'.")
            exit(1)
        if not args.length:
            print("Le paramètre --length est requis pour le mode 'generate'.")
            exit(1)

        
        model = load_model(args.model)
        print(f"Modèle n-gram chargé depuis {args.model}")
        with open(args.output_generation, 'w', encoding='utf8') as f:
            generation = generate_text(model, args.seed, args.length)
            f.write(generation)
            print(f"Texte généré : {generation[:100]}...")
    elif args.mode == "both" :
        if not args.input:
            print("Le paramètre --input est requis pour le mode 'both'.")
            exit(1)
        if not os.path.exists(args.input):
            print(f"Le fichier d'entrée '{args.input}' n'existe pas.")
            exit(1)
        if not args.output_model:
            print("Le paramètre --output-model est requis pour le mode 'both'.")
            exit(1)
        if not args.order:
            print("Le paramètre --order est requis pour le mode 'both'.")
            exit(1)
        if not args.output_generation:
            print("Le paramètre --output-generation est requis pour le mode 'both'.")
            exit(1)
        if not args.seed:
            print("Le paramètre --seed est requis pour le mode 'both'.")
            exit(1)
        if not args.length:
            print("Le paramètre --length est requis pour le mode 'both'.")
            exit(1)

        raw_text = read_corpus(args.input)
        clean_text = normalize_text(raw_text)
        print(f"Texte normalisé : {clean_text[:100]}...")  # Affiche les 100 premiers caractères du texte normalisé
        ngram_model = build_ngram_model(clean_text, args.order)
        output_file = save_model(ngram_model, args.output_model, compress=args.compress)
        model = load_model(output_file)
        print(f"Modèle n-gram chargé depuis {output_file}")
        with open(args.output_generation, 'w', encoding='utf8') as f:
            generation = generate_text(model, args.seed, args.length)
            f.write(generation)
            print(f"Texte généré : {generation}...")
    else :
        print("Mode non reconnu. Utilisez 'analyse', 'generate' ou 'both'.")
        exit(1)
