# FeGa experimental spectra

Two acquisitions are supplied:

| Directory prefix | Nominal applied field | Grid | Scan area | Height |
| --- | --- | --- | --- | --- |
| `2026-05-25-10-27-11` | 10 mT | 110 x 110 pixels | 7 x 7 micrometres | 150 nm |
| `2026-05-25-23-05-22` | 50 mT | 110 x 110 pixels | 7 x 7 micrometres | 150 nm |

Each directory ends in `-odmr_hardware` and contains:

```text
imageMeta.json
seq0/eval/odmr.h5
seq0/eval/odmr.txt
```

The HDF5 and text files are unchanged acquisition/converted data. The notebook
uses the existing converted text file; no conversion step is required.
Text columns include raw pixel indices, frequency samples and measured counts.
`helpers.load_df1` applies the same image-index mapping as the research workflow:
`i = j_raw`, `j = Ni_raw - 1 - i_raw`. The raw data files are not rotated or edited.

`imageMeta.json` preserves scientific acquisition settings. The `operator` and
`serial_number` fields were removed from the public copy; neither enters the
analysis. All other metadata are retained. No scalar-data, unrelated acquisition,
instrument sequence or software-installation files are needed by these notebooks.

The nominal applied fields label the two acquisitions. They are not inferred
from the fitted unsigned maps. The 50 mT image guides manual annotation of the
10 mT map and does not provide an independent quantitative reference field.
