| PE  | WS  | DM  | Heads  | Train Acc  |     Val Acc    |   Test Acc   |  Convergance Epoch | Dropout | Caveat|
|---  |:--------------------: |:-------------------------:    |:------------: |:----------------------:| :--------------------: |:---------------------------------------:      | :--------------------: | :--------------------: |  :--------------------: |
| Yes|      22|        128|4      | 98.44\% |     87.82%    |       86\% |   80    | 0.1 | max length as feature; add(Flatten), remove(Dense + dropout) |

**PE = Positional Encoding Layer**, **WS = Window Size**, **DM = d_model**