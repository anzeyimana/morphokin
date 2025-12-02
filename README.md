
# MorphoKIN: Kinyarwanda morphological analyzer and synthesizer

MorphoKIN, the morphological analyzer and generator for Kinyarwanda is a software package for analyzing Kinyarwanda text for natural language processing applications.
MorphoKIN implements core algorithms for Kinyarwanda morphological analysis, synthesis, syllabic tokenization and text normalization for text-to-speech applications.
These core algorithms are described in detail in the referenced scientific publications which we made in the previous years.

## 1. Installing MorphoKIN

### Minimum System Requirements

- x86_64 CPU
- 64 GB of System RAM
- 64 GB of Disk Storage
- Nvidia Drivers
- Nvidia GPU
- Docker
- NVIDIA Container Toolkit

This tutorial was tested on [AWS EC2](https://aws.amazon.com/ec2/) *g6e.4xlarge* instance with "Amazon/Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04) 20251101" AMI.

### 1.1. Get a free license for MorphoKIN

The free license is only allowed for academic and non-commercial use of the morphological analyzer/synthesizer.
Refer to the [Terms and Conditions](https://docs.google.com/document/d/17elFQbP4lR8uSufsU1NymObH_t2z0dy7sq78fbIMU7M/view) for the morphological analyzer/synthesizer.

To request a free license, fill in the registration form available at:
https://morphokin.kinlp.com/license/request
The form requests basic information about the user and their organization.
Once submitted, you will be required to verify the email address by clicking on the confirmation link sent to your email address.

Once approved, a free license file will be sent to your email address.

### 1.2. Download and install MorphoKIN docker image

MorphoKIN software is provided as a Docker image and is available for download from the following Google Drive link:
https://drive.google.com/file/d/1_4_R2FewU2zEvuYNKMx5L4b7T-Ao92kL/view
With the link, you can download it directly to your machine.

In order to download the image from a terminal (i.e. on a remote server), you need to use a Google OAuth token, which you can generate following the steps below:
1. Go to OAuth 2.0 Playground https://developers.google.com/oauthplayground/
2. In the Select the Scope box, paste https://www.googleapis.com/auth/drive.readonly
3. Click Authorize APIs and then Exchange authorization code for tokens
4. Copy the Access token
5. Run the following command in terminal, where ACCESS_TOKEN is the access token copied above:

```shell

curl -H "Authorization: Bearer ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1_4_R2FewU2zEvuYNKMx5L4b7T-Ao92kL?alt=media -o morphokin.tar.gz

```
Load the downloaded docker image into docker:

```shell

docker load --input morphokin.tar.gz

```

### 1.3. Download and extract MorphoKIN DATA package

MorphoKIN requires a data package that contains runtime data and various configuration files.
You can download the data package from the following Google Drive link:
https://drive.google.com/file/d/15XOrMbTktC86Ngq5HWnC4zeG_MAnjAgP/view

You can also download it from the terminal with an OAuth ACCESS_TOKEN (see the above section).

```shell

curl -H "Authorization: Bearer ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/15XOrMbTktC86Ngq5HWnC4zeG_MAnjAgP?alt=media -o MORPHODATA.tar.gz

```
Extract the data package in local directory named MORPHODATA

```shell

gunzip -c MORPHODATA.tar.gz | tar x

```

## 2. Using MorphoKIN from shell terminal

### 2.1 Verify the free license validity

The following example assume you have downloaded your free license file (e.g. KINLP_LICENSE_FILE.dat) inside */home/ubuntu/MORPHODATA/licenses/* directory.
```shell

# Ensure you have the free license file, e.g.
# cp KINLP_LICENSE_FILE.dat /home/ubuntu/MORPHODATA/licenses/

docker run --rm -v /home/ubuntu/MORPHODATA:/MORPHODATA --gpus all -it morphokin:latest bash

morphokin --morphokin_working_dir /MORPHODATA --task LICENSE --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat  --ca_roots_pem_file /MORPHODATA/data/roots.pem

```

### 2.2 Morphological parsing of Kinyarwanda text

MorphoKIN parsing generates a tab-separated list of word parses where each word contains parse ids(tab-separated) and surface form separated by space.
Parse ids contain the following fields: *lm_stem_id,lm_morph_id,pos_tag_id,stem_id,len(extra_stem_token_ids),extra_stem_token_ids,len(affix_ids),affix_ids* which we describe below:
- lm_stem_id: common stem id, this is an index into a list of common Kinyarwanda stems [0 - 9999]
- lm_morph_id: morphological tag id, this is an index into a list of common Kinyarwanda morphological parses [0 - 24121]
- pos_tag_id: part-of-speech tag id [0 - 156]
- stem_id: fine-grained stem id or sub-word token id [0 - 35496]
- len(extra_stem_token_ids): number of extra sub-word tokens
- extra_stem_token_ids: extra sub-word ids [0 - 35496]
- len(affix_ids): number of affixes
- affix_ids: affix ids [0 - 406]

Example parse: 
*438,11,18,336,0,3,17,31,7 Umuhinzi	20,1053,15,62,0,5,20,37,5,11,6 arasabwa	201,1920,27,14,0,3,150,201,170 iki	7,24029,64,8,0,0 mu	33,21,5,59,0,3,24,5,6 gukomeza	1568,21,5,1465,0,3,24,5,6 gusigasira	807,24,18,1135,0,3,10,50,7 ibishanga	542,2749,15,785,0,6,27,9,5,54,11,8 byatunganijwe	53,24065,100,130,0,0 ?*

Parse single text file:

```shell

docker run --rm -v /home/ubuntu/MORPHODATA:/MORPHODATA --gpus all -it morphokin:latest bash

morphokin --morphokin_working_dir /MORPHODATA --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf  --task PTF --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat  --ca_roots_pem_file /MORPHODATA/data/roots.pem --input_file /sample_corpus.txt --output_file /preparsed_sample_corpus.txt

```

Batch parse a list of text files in the same directory:

```shell

docker run --rm -v /home/ubuntu/MORPHODATA:/MORPHODATA --gpus all -it morphokin:latest bash

morphokin --morphokin_working_dir /MORPHODATA --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf  --task BTF --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat  --ca_roots_pem_file /MORPHODATA/data/roots.pem --batch_files_dir / --batch_files_list sample_corpus.txt,another_corpus.txt

```

### 2.3 Run MorphoKIN server on Unix domain socket

This is for morphological analysis and synthesis by external programs

```shell

docker run --rm -v /home/ubuntu/MORPHODATA:/MORPHODATA --gpus all -it morphokin:latest bash

nohup morphokin --morphokin_working_dir /MORPHODATA --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf  --task RMS --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat  --ca_roots_pem_file /MORPHODATA/data/roots.pem --morpho_socket /MORPHODATA/run/morpho.sock  &>> rms.log &

```

## 3. Python usage examples

These examples need to MorphoKIN to be running on unix domain socket (See section 2.3 above)

```shell

docker run -d -v /home/ubuntu/MORPHODATA:/MORPHODATA \
  --gpus all morphokin:latest morphokin \
  --morphokin_working_dir /MORPHODATA \
  --morphokin_config_file /MORPHODATA/data/analysis_config_file.conf  \
  --task RMS \
  --kinlp_license /MORPHODATA/licenses/KINLP_LICENSE_FILE.dat  \
  --ca_roots_pem_file /MORPHODATA/data/roots.pem \
  --morpho_socket /MORPHODATA/run/morpho.sock

```

You will have to wait for MorphoKIN socket server to be ready by monitoring the container logs.

```shell

docker container ls

docker logs -f <CONTAINER ID>

# MorphoKIN server is ready once you see a message like this: MorphoKin server listening on UNIX SOCKET: /MORPHODATA/run/morpho.sock

```

### 3.1 Morphological analysis and POS Tagging
```python

from morphokin.data.kinyarwanda import parse_text_to_morpho_sentence, init_morphokin_socket, ParsedFlexSentence, pos_tag_view
from morphokin.data.uds_client import UnixSocketClient

uds_client: UnixSocketClient = init_morphokin_socket(connect_to_flexkin_socket=True, sock_file='/home/ubuntu/MORPHODATA/run/morpho.sock')
text: str = "Ikiremwamuntu cyose kivukana umudendezo kandi kingana mu cyubahiro n'uburenganzira. Gifite ubushobozi bwo gutekereza n'umutimanama kandi kigomba gukorera bagenzi bacyo mu mwuka wa kivandimwe."
parsed: ParsedFlexSentence = parse_text_to_morpho_sentence(uds_client, text)

print(parsed.to_parsed_format())

print('\n'.join([pos_tag_view(t.pos_tag_id) for t in parsed.tokens]))

```

### 3.2 Text Normalization (i.e. for TTS)
```python
from morphokin.data.kinyarwanda import parse_text_to_morpho_sentence, pos_tag_view, norm_text, init_morphokin_socket

uds_client = init_morphokin_socket(connect_to_flexkin_socket=True, sock_file='/home/ubuntu/MORPHODATA/run/morpho.sock')
text = "Iyi kipe yatwaye irushanwa rya 2023, yakurikiwe n’iy’Umudage Lukas Baum watwaye Cape Epic mu 2022, kuri ubu uri gukina afatanyije n’Umunya-Kenya Dan Kiptala aho bo bakoresheje iminota 26 n’amasegonda 48."
normalized_text = norm_text(text)

print(normalized_text)


```

### 3.3 Syllabic Tokenization (i.e. for TTS)
```python
from morphokin.data.syllable_vocab import text_to_id_sequence

text = "Ikiremwamuntu cyose kivukana umudendezo kandi kingana mu cyubahiro n'uburenganzira. Gifite ubushobozi bwo gutekereza n'umutimanama kandi kigomba gukorera bagenzi bacyo mu mwuka wa kivandimwe."
id_seq = text_to_id_sequence(text)

print(id_seq)


```

### 3.4 Morphological synthesis
```python

# TBD

```


## References

[1] Antoine Nzeyimana. 2020. Morphological disambiguation from stemming data. In Proceedings of the 28th International Conference on Computational Linguistics, pages 4649–4660, Barcelona, Spain (Online). International Committee on Computational Linguistics.

[2] Antoine Nzeyimana and Andre Niyongabo Rubungo. 2022. KinyaBERT: a Morphology-aware Kinyarwanda Language Model. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5347–5363, Dublin, Ireland. Association for Computational Linguistics.

[3] Antoine Nzeyimana. 2023. KINLP at SemEval-2023 Task 12: Kinyarwanda Tweet Sentiment Analysis. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 718–723, Toronto, Canada. Association for Computational Linguistics.

[4] Antoine Nzeyimana. 2024. Low-resource neural machine translation with morphological modeling. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 182–195, Mexico City, Mexico. Association for Computational Linguistics.

[5] Antoine Nzeyimana, and Andre Niyongabo Rubungo. 2025. KinyaColBERT: A Lexically Grounded Retrieval Model for Low-Resource Retrieval-Augmented Generation. arXiv preprint arXiv:2507.03241.
