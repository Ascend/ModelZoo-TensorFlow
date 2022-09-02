# Valse Movements

| PE  | WS  | DM  | Heads  | Train Acc  |     Val Acc    |   Test Acc   |  Convergance Epoch | Dropout | Caveat|
|---  |:--------------------: |:-------------------------:    |:------------: |:----------------------:| :--------------------: |:---------------------------------------:      | :--------------------: | :--------------------: |  :--------------------: |
| Yes|      32|        128|4      | 96.17\% |     85.97\%    |       65\% |   58    |
| Yes|      32|        128|4      | 90.06\% |     85.56\%    |       70\% |   57    | 0.2 | length as attribute |
| Yes|      32|        128|4      | 94.28\% |     86.46\%    |       72\% |   61    | 0.1 | length as attribute |
| Yes|      32|        128|4      | 96.51\% |     89.84\%    |       76\% |   79    | 0.2 | max length as feature; add(Flatten), remove(Dense + dropout) |
| Yes|      32|        128|4      | **96.10\%** |     **89.70\%**    |       **80\%** |   47    | 0.1 | max length as feature; remove sensor attention; add(Flatten), remove(Dense + dropout) |
| Yes|      32|        128|4      | 97.90\% |     89.37\%    |       77\% |   62    | 0.1 | max length as feature; add(Flatten), remove(Dense + dropout) |
| Yes|      32|        128|4      | 95.27\% |     87.80\%    |       72\% |   46    | 0.1 |length as feature; add(Flatten), remove(Dense + dropout) |
| Yes|      32|        128|4      | 94.62\% |     81.70\%    |       67\% |   56    | 0.1 | |
| Yes|      32|        128|4      | 96.73\% |     82.79\%    |       66\% |   49    | 0.1 | add(Flatten), remove(Dense + dropout)|
| Yes|      32|        256|8      | 86.33\% |     87.11\%    |       62\% |   114   |
| Yes|      49|        256|8      | 96.78\% |     83.92\%    |       65\% |   37   |  0.1 | add(Flatten), remove(Dense + dropout) |
| No |      49|        128|4      | 99.73\% |     88.29\%    |       67\% |   38    |  0.1 | add(Flatten), remove(Dense + dropout) |
| No |      49|        128|4      | 96.26\% |     88.62\%    |       70\% |   47    |  0.2 | length as feature |
| No |      49|        128|4      | 98.02\% |     89.63\%    |       72\% |   62    |  0.1 | length as feature |
| No |      49|        128|4      | 99.85\% |    88.57\%    |       79\% |   54    |  0.1 | max length as feature; remove sensor attention; add(Flatten), remove(Dense + dropout)|
| No |      49|        128|4      | 99.26\%** |    90.07\%    |       72\% |   48    |  0.1 | length as feature; add(Flatten), remove(Dense + dropout)|
| No |      49|        128|4      | 99.89\% |     88.62\%    |       68\% |   67    |  0.1 |
| No |      49|        128|4      | 98.63\% |87.11\%| 70\% |   68    |
| No |      15|        128|4      | 95.28\% |     85.26\%    |       68\% |   76    |
| No |      32|        128|4      | 94.24\% |     85.00\%    |       68\% |   46    |
| No |      49|        256|4      | 95.64\% |     86.19\%    |       69\% |   67    |
| No |      49|        256|8      | 95.89\% |     86.94\%    |       68\% |   54    |
| No |      49|        128|8      | 97.45\% |     87.07\%    |       65\% |   50    |
| No |      49|        64 |2      | 93.46\% |     85.39\%    |       67\% |   46    |
| No |      49|        32 |2      | 90.28\% |     84.34\%    |       66\% |   46    |
| No |      49|        128|4      | 95.80\% |     86.15\%    |       67\% |   60    |  0.3 |



**PE = Positional Encoding Layer**, **WS = Window Size**, **DM = d_model**
