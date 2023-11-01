import functools
import pickle

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from hardpicks.analysis.fbp.report.path_constants import pickles_directory
from hardpicks import FBP_DATA_DIR, FBP_BAD_GATHERS_DIR
from hardpicks.data.fbp.auto_pick_parser import AutoPickParser
from hardpicks.data.fbp.collate import fbp_batch_collate
from hardpicks.data.fbp.constants import BAD_FIRST_BREAK_PICK_INDEX
from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset
from hardpicks.data.fbp.site_info import get_site_info_by_name
from hardpicks.metrics.fbp.preconfigured_evaluator import get_preconfigured_regression_evaluator

lalor_site_dict = dict(site_name='Lalor',
                       shot_identifier_key='shot_id',
                       file_names=["Lalor_autopicks_th0_3.pic",
                                   "Lalor_autopicks_th1_0.pic"])

brunswick_site_dict = dict(site_name='Brunswick',
                           shot_identifier_key='shot_id',
                           file_names=["Brunswick_auto_picks_for_Mila_th0_3.pic",
                                       "Brunswick_auto_picks_for_Mila_th1_0.pic"])

# matagami_site_dict = dict(site_name='Matagami',
#                           shot_identifier_key='shot_id',
#                           file_names=["MatagamiWest3D_part_autopicks_th0_3.pic",
#                                       "MatagamiWest3D_part_autopicks_th1_0.pic"])

sudbury_site_dict = dict(site_name='Sudbury',
                         shot_identifier_key='shot_number',
                         file_names=["Sudbury_picks_forMila_max_trough_th0_3.pic",
                                     "Sudbury_picks_forMila_max_trough_th1_0.pic"])

halfmile_site_dict = dict(site_name='Halfmile',
                          shot_identifier_key='shot_id',
                          file_names=["Halfmile3D_auotpicks_th0_3.pic",
                                      "Halfmile3D_auotpicks_th1_0.pic"])

list_site_dicts = [
    sudbury_site_dict,
    lalor_site_dict,
    brunswick_site_dict,
    # matagami_site_dict,
    halfmile_site_dict,
]

autopicks_dir = FBP_DATA_DIR.joinpath("autopicks/various_thresholds/")

pickle_dump_path = pickles_directory.joinpath("baseline_summaries.pkl")

rejected_gather_yaml_path = FBP_BAD_GATHERS_DIR.joinpath("bad-gather-ids_combined.yaml")

if __name__ == '__main__':

    all_summaries_dict = dict()
    for site_dict in list_site_dicts:
        site_name = site_dict['site_name']
        shot_identifier_key = site_dict['shot_identifier_key']

        list_pick_file_names = site_dict['file_names']

        site_info = get_site_info_by_name(site_name)

        raw_dataset = create_shot_line_gather_dataset(site_info["processed_hdf5_path"],
                                                      site_name,
                                                      site_info["receiver_id_digit_count"],
                                                      site_info["first_break_field_name"],
                                                      provide_offset_dists=True
                                                      )
        sample_rate_in_milliseconds = raw_dataset.samp_rate / 1000.

        clean_dataset = ShotLineGatherCleaner(raw_dataset,
                                              rejected_gather_yaml_path=rejected_gather_yaml_path)
        collate_fn = functools.partial(
            fbp_batch_collate,
            pad_to_nearest_pow2=True
        )

        dataloader_parameters = dict(
            dataset=clean_dataset,
            num_workers=0,
            collate_fn=collate_fn,
            batch_size=64,
            shuffle=False
        )

        data_loader = DataLoader(**dataloader_parameters)

        for pick_filename in list_pick_file_names:
            evaluator = get_preconfigured_regression_evaluator()
            autopicks_file_path = autopicks_dir.joinpath(pick_filename)
            auto_pick_parser = AutoPickParser(site_name,
                                              shot_identifier_key,
                                              autopicks_file_path,
                                              sample_rate_in_milliseconds)
            number_of_ground_truth_picks = 0
            number_of_relevant_predicted_picks = 0
            for batch_idx, batch in tqdm(enumerate(data_loader), 'BATCH'):
                raw_preds = auto_pick_parser.get_raw_preds(batch)
                evaluator.ingest(batch, batch_idx, raw_preds)

                good_ground_truth_first_break_masks = ~batch['bad_first_breaks_mask'].numpy()
                predictions_when_good_ground_truth = raw_preds.numpy()[good_ground_truth_first_break_masks]
                number_of_ground_truth_picks += np.sum(good_ground_truth_first_break_masks)
                number_of_relevant_predicted_picks += np.sum(
                    predictions_when_good_ground_truth > BAD_FIRST_BREAK_PICK_INDEX)

            ratio = number_of_relevant_predicted_picks / number_of_ground_truth_picks
            summary = evaluator.summarize()
            summary['GatherCoverage'] = ratio
            print(pick_filename)
            print(summary)
            all_summaries_dict[pick_filename] = summary

    with open(pickle_dump_path, "wb") as fd:
        pickle.dump(all_summaries_dict, fd)
