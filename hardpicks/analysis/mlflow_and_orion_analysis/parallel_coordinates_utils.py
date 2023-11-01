import pandas as pd
from plotly import graph_objs as go

inch_to_pixel = 96
parallel_plot_layout = go.Layout(
    width=7.20 * inch_to_pixel,
    height=4.45 * inch_to_pixel,
    font_family="Serif",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    hovermode="closest",
    legend=dict(yanchor="top", y=0.85, xanchor="right", x=0.85),
    margin=dict(l=110, r=5, t=50, b=5),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font={"color": "black", "size": 12},
)


def process_categorical_series(input_series, categories=None):
    """Turn input series into pandas categoricals."""
    if categories:
        processed_series = pd.Categorical(input_series, categories=categories)
    else:
        processed_series = pd.Categorical(input_series)
    return processed_series


def process_decoder_block_channels_series(decoder_block_channels_series):
    """Turn decoder block channels input series into pandas categoricals."""
    # explicitly sort the decoder categories so we don't end up with lexical sorting
    decoder_categories = [
        "[256, 128, 64, 32, 16]",
        "[512, 256, 128, 64, 32]",
        "[1024, 512, 256, 128, 64]",
    ]
    return process_categorical_series(
        decoder_block_channels_series, categories=decoder_categories
    )


def process_loss_type_series(loss_type_series):
    """Turn loss type input series into pandas categoricals."""
    conversion_dictionary = {"dice": "Dice", "crossentropy": "Cross-entropy"}
    categories = list(conversion_dictionary.values())
    return process_categorical_series(
        loss_type_series.replace(conversion_dictionary), categories=categories
    )


def process_class_count_series(class_count_series):
    """Turn class count input series into pandas categoricals."""
    conversion_dictionary = {1: "Binary", 3: "Ternary"}
    categories = list(conversion_dictionary.values())
    return process_categorical_series(
        class_count_series.replace(conversion_dictionary), categories=categories
    )


def process_encoder_type_series(encoder_type_series):
    """Turn encoder type input series into pandas categoricals."""
    conversion_dictionary = {
        "resnet18": "ResNet18",
        "resnet34": "ResNet34",
        "timm-efficientnet-b0": "EffNetB0",
        "timm-efficientnet-b2": "EffNetB2",
        "timm-efficientnet-b4": "EffNetB4",
    }
    encoder_categories = list(conversion_dictionary.values())
    return process_categorical_series(
        encoder_type_series.replace(conversion_dictionary),
        categories=encoder_categories,
    )


def process_orion_dataframe(orion_df, new_objective_name):
    """Manipulate the orion dataframe so that it is ready for plotting/tabulating."""
    column_rename_dict = {
        "id": "id",
        "objective": "objective",
        "/decoder_block_channels": "Decoder Blocks",
        "/loss_type": "Loss Type",
        "/optimizer_params/lr": "Learning Rate",
        "/scheduler_params/step_size": "Scheduler Step",
        "/segm_class_count": "Class Setup",
        "/unet_encoder_type": "Encoder Type",
    }

    categorical_columns_processing_dict = {
        "Decoder Blocks": process_decoder_block_channels_series,
        "Class Setup": process_class_count_series,
        "Scheduler Step": process_categorical_series,
        "Loss Type": process_loss_type_series,
        "Encoder Type": process_encoder_type_series,
    }

    completed_mask = orion_df["status"] == "completed"

    desired_columns = list(column_rename_dict.keys())
    df = (
        orion_df[completed_mask]
        .sort_values(by="objective")[desired_columns]
        .rename(columns=column_rename_dict)
        .set_index("id")
    )
    df[new_objective_name] = -df["objective"]
    del df["objective"]

    for (
        categorical_column_name,
        series_processor,
    ) in categorical_columns_processing_dict.items():
        df[categorical_column_name] = series_processor(df[categorical_column_name])

    return df
