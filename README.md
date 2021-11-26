# HardPicks Benchmark

This repository provides links, metadata, and other useful information regarding the seismic survey
datasets that were introduced in the NeurIPS 2021 ML4PS Workshop paper titled "A Multi-Survey Dataset
and Benchmark for First Break Picking in Hard Rock Seismic Exploration". Additional information may
be added later following other publications using this data.

## Data

Before downloading any data, make sure you read and understand the licensing terms below.

### Brunswick and Halfmile Lake 3D Surveys License

Mila and Natural Resources Canada have obtained licences from Glencore Canada Corporation and Trevali
Mining Corporation to distribute field seismic data from the Brunswick 3D and Halfmile Lake 3D seismic
surveys, respectively, under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](
https://creativecommons.org/licenses/by/4.0/). These datasets are in the Hierarchical Data Format
(HDF5) and have first arrival labels included in trace headers.

### Lalor and Sudbury 3D Surveys License

The Lalor 3D and Sudbury 3D seismic data are distributed under the [Open Government Licence – Canada]( https://open.canada.ca/en/open-government-licence-canada). Canada grants to the licensee a non-exclusive,
fully paid, royalty-free right and licence to exercise all intellectual property rights in the data. This
includes the right to use, incorporate, sublicense (with further right of sublicensing), modify, improve,
further develop, and distribute the Data; and to manufacture or distribute derivative products.

The formatting of these datasets is similar to the other two.

Please use the following attribution statement wherever applicable:

    Contains information licensed under the Open Government Licence – Canada.

### Download links

The HDF5 files are hosted on AWS, and can be downloaded directly:
 - [Brunswick](https://d3sakqnghgsk6x.cloudfront.net/Brunswick_3D/Brunswick_orig_1500ms_V2.hdf5.xz)
 - [Halfmile Lake](https://d3sakqnghgsk6x.cloudfront.net/Halfmile_3D/Halfmile3D_add_geom_sorted.hdf5.xz)
 - [Lalor](https://d3sakqnghgsk6x.cloudfront.net/Lalor_3D/Lalor_raw_z_1500ms_norp_geom_v3.hdf5.xz)
 - [Sudbury](https://d3sakqnghgsk6x.cloudfront.net/Sudbury_3D/Sudbury3D_all_shots_2s.hdf.xz)

### Data loading demo

We demonstrate how to parse and display the data in [this notebook](./fbp_data_loading_demo.ipynb).

## Acknowledgements

We thanks Glencore Canada Corporation and Trevali Mining Corporation for providing access and allowing us
to include and distribute the Brunswick 3D and Halfmile 3D seismic data as part of this benchmark dataset.
We also thank E. Adam, S. Cheraghi, and A. Malehmir for providing first breaks for the Brunswick, Halfmile,
and Sudbury data.

## References

P.-L. St-Charles, B. Rousseau, J. Ghosn, J.-P. Nantel, G. Bellefleur, E. Schetselaar;
"A Multi-Survey Dataset and Benchmark for First Break Picking in Hard Rock Seismic Exploration"
in Proc. Neurips 2021 Workshop on Machine Learning for the Physical Sciences (ML4PS), 2021.
