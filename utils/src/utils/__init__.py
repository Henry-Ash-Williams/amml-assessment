from .data import ImageSequenceDataset

SEQUENCE_LENGTH = 16
TRAIN_SEQUENCE_LENGTH = 14
TEST_SEQUENCE_LENGTH = 2
N_SEQUENCES = 400
IMAGE_SHAPE = (36, 36, 1)

n6_train = ImageSequenceDataset(
    "/Users/henrywilliams/Documents/uni/amml/assessment-2/data/n6-train.csv",
    14,
    IMAGE_SHAPE,
)
n6_test = ImageSequenceDataset(
    "/Users/henrywilliams/Documents/uni/amml/assessment-2/data/n6-test.csv",
    2,
    IMAGE_SHAPE,
)

n3_train = ImageSequenceDataset(
    "/Users/henrywilliams/Documents/uni/amml/assessment-2/data/n3-train.csv",
    14,
    IMAGE_SHAPE,
)
n3_test = ImageSequenceDataset(
    "/Users/henrywilliams/Documents/uni/amml/assessment-2/data/n3-test.csv",
    2,
    IMAGE_SHAPE,
)
