"""Where the tooling reads and writes its files. See TRAINING.md.

training-data/ holds the photos you add (not in source control):
  decks/              decks laid out in a grid, every dot showing (textures for the training boards)
  games/              finished games (to test a model); games/counts.json has known counts
out/ holds everything the scripts make; it can be deleted at any time.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DECKS = os.path.join(REPO_ROOT, "training-data", "decks")
GAMES = os.path.join(REPO_ROOT, "training-data", "games")
COUNTS = os.path.join(GAMES, "counts.json")

# The shipped model's weights (in source control); training starts from it, export replaces it.
MODEL = os.path.join(REPO_ROOT, "tooling", "detector", "tactanet.pt")

OUT = os.path.join(REPO_ROOT, "out")
CARD_ASSETS = os.path.join(OUT, "card-assets")     # recolourable card textures + deck icons
BOARDS = os.path.join(OUT, "boards")               # synthetic boards, one folder per split
RUNS = os.path.join(OUT, "runs")                   # one folder per training run
TEST_CASES = os.path.join(OUT, "test-cases")
SCORING_CASES = os.path.join(TEST_CASES, "scoring")  # npm test: TypeScript scoring vs Python
PARITY_CASE = os.path.join(TEST_CASES, "parity")     # npm test: TypeScript engine vs PyTorch

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".webp")


def photos(folder):
    """The image files in a folder, sorted by name (case-insensitive)."""
    if not os.path.isdir(folder):
        return []
    names = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_TYPES)]
    return [os.path.join(folder, f) for f in sorted(names, key=str.lower)]
