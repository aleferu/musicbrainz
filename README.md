# MUSYNERGY

Code for MUSYNERGY: A Framework for Music Collaboration Discovery Based on Neural Networks and Graph Analysis.

Article: https://www.sciencedirect.com/science/article/pii/S1875952125001132

## .env file

A .env file is needed in order to configure the database connection. An example .env file is provided. Modify its contents to satisfy your needs.

## Important Directories

- `mb_extraction` explains how we got the information out of MB.
- `pre_lfm` explains how we prepared before getting the information out of LFM and before using neo4j.
- `lfm` explains how we got the information out of LFM and into neo4j.
- `post_data` explains how we got the dataset for the MLP and has some scripts and notebooks to compute and graph some stats.
- `pyg_experiments` explains how we've trained the models.
- `tiktok` extracts data from TikTok's Research API and loads it into Neo4j.
- `tiktok_pyg` trains the TikTok models with all the artists with an associated TikTok account.
- `intiktok_pyg` trains the TikTok models with only the artists with a successful data extraction from TikTok.

## Dates

- Musicbrainz data: `2024-10-31 21:00:49.130771+00`
- LastFM data: `2024-11-05` to `2024-11-23`.
- TikTok data: `2025-10-13` to `2025-11-25`.

## Dataset

`schema.md`are dedicated to explain our dataset. With these two files you can learn about the root graph schema and check out some graphs and statistics.

## Dependencies - conda

An `environment.yaml` is provided.

```bash
conda env create -f environment.yaml
```

Or with custom name.

```bash
conda env create -n customenvname -f environment.yaml
```

If pip fails install `pyg-lib` after everything else is done.

> [!IMPORTANT]  
> Since then, pytorch decided to drop conda support, so the better way of making everything work is to get a cuda-working environment running with pytorch and install what you need based on what you want to run. [Check this link](https://github.com/pytorch/pytorch/issues/138506).

## PostgreSQL (MusicBrainz) database installation

I followed [this link](https://musicbrainz.org/doc/MusicBrainz_Server/Setup), section **Setup from source code**. In theory you can install the database without the web server but I didn't want to mess up by skipping a crucial step by accident. The web server installation is very fast, unlike the database installation, so I recommend doing it that way.

## Neo4j database installation

https://neo4j.com/docs/operations-manual/current/installation/linux/debian/


Configuration (added at the start of `/etc/neo4j/neo4j.conf`):

```
server.memory.heap.max_size=11G
server.memory.heap.initial_size=5G
dbms.security.auth_enabled=false
dbms.usage_report.enabled=false
dbms.security.procedures.unrestricted=apoc.*,gds.*
dbms.security.procedures.allowlist=apoc.*,gds.*
dbms.memory.transaction.total.max=10g
```

Configuration added to `/etc/neo4j/apoc.conf` (might need to create file):

```
apoc.import.file.enabled=true
```

`APOC` should already be downloaded somewhere in `/var/lib/neo4j/`, but you need to download and install `GDS` from [this link](https://neo4j.com/deployment-center/#gds-tab) (also check [version compatibilities](https://neo4j.com/docs/graph-data-science/current/installation/supported-neo4j-versions/) and [instructions](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-server/)).

Then just `sudo neo4j start/stop/restart...` to run.

## How to cite

```elixir
@article{FernandezSanchez2025,
  author = {A. Fernandez-Sanchez and Pedro J. Navarro and F. Terroso-Saenz},
  title = {{MUSYNERGY: A Framework for Music Collaboration Discovery Based on Neural Networks and Graph Analysis}},
  year = {2025},
  journal = {Entertainment Computing},
  doi = {10.1016/j.entcom.2025.101033},
}
```
