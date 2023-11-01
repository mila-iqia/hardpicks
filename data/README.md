This folder contains raw and preprocessed versions of the data used in experiments,
task-specific data configuration files (e.g. bad gather lists, fold configurations, ...),
and artifacts resulting from data analysis. Some of the files in here might be quite heavy
and involved in lots of disk I/O operations, so it might be best to create symbolic links
to the files (or to their parent folders) and place them on fast partitions.

This folder should not contain scripts, as they become hidden away or deleted across other
folders that are replaced by symbolic links or periodically cleaned up. Data analysis
scripts should instead be placed somewhere in the `analysis` package of the source
directory.

