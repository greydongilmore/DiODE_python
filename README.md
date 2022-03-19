<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
        <li><a href="#repository-structure">Repository Structure</a></li>
      </ul>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
      <ul>
        <li><a href="#Publications">Publications</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

These python scripts replicate the Matlab implementation developed by Till Dembek in a program name DiODe. The original code can be found <a href="https://github.com/Till-Dembek/DiODe_Standalone" target="_blank"><strong>here</strong></a>.

Information about the algorithm can be found <a href="https://dx.doi.org/10.13140/RG.2.2.22417.76647" target="_blank"><strong>here</strong></a>.

Not for commercial use! Only fully validated for Boston Scientific directional leads.

### Built With

* Python version: 3.9


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. In a terminal, clone the repo by running:

    ```sh
    git clone https://github.com/greydongilmore/DiODE_python.git
    ```

2. Change into the project directory (update path to reflect where you stored this project directory):

    ```sh
    cd /home/user/Documents/Github/DiODE_python
    ```

3. Install the required Python packages:

    ```sh
    python -m pip install -r requirements.txt
    ```


<!-- USAGE EXAMPLES -->
## Usage

1. In a terminal, move into the project directory
     ```sh
     cd /home/user/Documents/Github/DiODE_python
     ```

2. Run the following to execute the epoch script:
    ```sh
    DiODE_python main.py -i "full/path/to/nifti_file" -e 'electrode_name' -rh -14.95,24.64,52.19 -rt -21.24,8.84,72.19 -lh 5.43,25.12,52.28 -lt 15.08,12.11,71.03
    ```
    or
    ```sh
    DiODE_python main.py -i "full/path/to/nifti_file" -e 'electrode_name' -fcsv 'full/path/to/fcsv_coords'
    ```

  * **-i:** full directory path to the postoperative CT scan with electrodes (in `nii.gz`)
  * **-e:** the name of the electrode model (for acceptable electrodes see [diode_elspec.py](./diode_elspec.py))
  * **-rh/-lh:** the right/left RAS coordinates for the bottom of the most distal contact
  * **-rh/-lh:** the right/left RAS coordinates for another point along the electrode
  * **-fcsv:** optionally, you can provide an fcsv file containing the coordinates (these files are generated in 3D Slicer where it might be easier to identify the required coordinates and store them as a Markups file in 3D Slicer)


### Repository Structure

The repository has the following scheme:
```
├── README.md
├── LICENSE
├── requirements.txt            # python libraries required to run these scripts
├── data
   ├── *nii.gz                 # sample CT images containing B.Sci directional electrodes
|   ├── *fcsv                   # associated RAS coordinates for the head/tail of each electrode
|   └── imgs                    # default output colder for generated figures
├── main.py                    # potentially cleaned-up version
├── diode_auto.py               # first pass conversion from Matlab (less efficient but still runs)
├── read_fcsv.m                 # Matlab function to read fcsv files, could be used in the Matlab implementation
├── diode_elspec.py             # electrode specification `.py` that writes the JSON file (can use this to add additional electrodes)
└── diode_elspec.json           # electrode specifications for the accepted electrode models
    
```
<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the GNU License. See [`LICENSE`](LICENSE)for more information.


<!-- CONTACT -->
## Contact

IP and any questions:
Till Dembek - [@DembekTill](https://twitter.com/dembektill) - till.dembek@uk-koeln.de

Conversion to Python:
Greydon Gilmore - [@GilmoreGreydon](https://twitter.com/GilmoreGreydon) - greydon.gilmore@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

### Publications

* https://dx.doi.org/10.1016/j.parkreldis.2019.08.017
* https://dx.doi.org/10.1159/000494738
* https://dx.doi.org/10.1002/mp.12424


* README format was adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/greydongilmore/DiODE_python.svg?style=for-the-badge
[contributors-url]: https://github.com/greydongilmore/DiODE_python/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/greydongilmore/DiODE_python.svg?style=for-the-badge
[forks-url]: https://github.com/greydongilmore/DiODE_python/network/members
[stars-shield]: https://img.shields.io/github/stars/greydongilmore/DiODE_python.svg?style=for-the-badge
[stars-url]: https://github.com/greydongilmore/DiODE_python/stargazers
[issues-shield]: https://img.shields.io/github/issues/greydongilmore/DiODE_python.svg?style=for-the-badge
[issues-url]: https://github.com/greydongilmore/DiODE_python/issues
[license-shield]: https://img.shields.io/github/license/greydongilmore/ocr-pdf.svg?style=for-the-badge
[license-url]: https://github.com/greydongilmore/ocr-pdf/blob/master/LICENSE.txt
