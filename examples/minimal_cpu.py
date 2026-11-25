"""Fit and evaluate a small SOE embedding on CPU."""

from soe import SOE, evaluate_embedding


def main() -> None:
    # These disjoint triplet sets describe six objects ordered along a line.
    training_triplets = [
        (0, 1, 4), (0, 2, 5), (1, 2, 5), (1, 0, 4),
        (2, 1, 5), (2, 3, 0), (3, 4, 0), (3, 2, 5),
        (4, 3, 0), (4, 5, 1), (5, 4, 1), (5, 3, 0),
    ]
    validation_triplets = [
        (0, 1, 3), (1, 3, 5), (2, 3, 5), (3, 4, 1), (4, 5, 2), (5, 4, 2),
    ]
    test_triplets = [
        (0, 2, 4), (1, 2, 4), (2, 1, 4), (3, 4, 5), (4, 3, 2), (5, 3, 1),
    ]

    model = SOE(
        n_components=2,
        learning_rate=0.1,
        batch_size=4,
        max_epochs=30,
        random_state=42,
        shuffle_random_state=43,
    )
    embedding = model.fit_transform(
        training_triplets, n_objects=6, validation_triplets=validation_triplets
    )
    checkpoint = model.best_checkpoint_
    print(f"Device: {model.device_}; batch size: {model.batch_size_}")
    print(
        f"Selected checkpoint: epoch {checkpoint.epoch}, "
        f"update {checkpoint.optimizer_step}, "
        f"{model.selection_source_} loss {checkpoint.selection_loss:.4f}"
    )
    print(f"Embedding shape: {embedding.shape}")
    print(embedding)

    test_metrics = evaluate_embedding(
        embedding,
        test_triplets,
        margin=model.margin,
        reduction=model.reduction,
        ordinal_mode=model.ordinal_mode,
    )
    print(f"Held-out ordinal error: {test_metrics.ordinal_error:.1%}")


if __name__ == "__main__":
    main()
