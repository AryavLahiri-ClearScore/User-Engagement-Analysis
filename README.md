User Types: 

High Engagement: 36

Medium Engagement: 34

Low Engagement: 30

Content preference

High Engagement users: credit_cards, insights, loans, protect, drivescore, improve 

Medium Engagement users: insights, loans, drivescore, protect, credit_cards

Low Engagement users: improve,insights, drivescore, credit_cards, protect, loans 

Most common content types based off interactions:
1. insights
2. improve
3. credit_cards
4. protect
5. drive score
6. loans

Based of click rate:
1. credit_cards
2. drivescore
3. improve
4. loans
5. protect
6. insights

As somewhat expected average time viewed was highest for high engagement and lowest for low engagement


DAY 1: EXPLORATORY DATA ANALYSIS
============================================================
Dataset shape: (1000, 6)
Date range: 2025-06-09 to 2025-07-09
Number of unique users: 100
Number of unique content pieces: 50

1. MOST COMMON CONTENT TYPES:
   insights: 177 interactions (17.7%)
   improve: 174 interactions (17.4%)
   credit_cards: 167 interactions (16.7%)
   protect: 162 interactions (16.2%)
   drivescore: 160 interactions (16.0%)
   loans: 160 interactions (16.0%)

2. TIME SPENT ON PAGES:
   Average time by content type:
   credit_cards: 30.4s avg, 22.7s median
   drivescore: 27.8s avg, 20.4s median
   improve: 28.5s avg, 16.8s median
   insights: 28.5s avg, 18.1s median
   loans: 26.6s avg, 19.2s median
   protect: 33.3s avg, 24.2s median

3. CLICK RATES BY CONTENT TYPE:
   credit_cards: 36.5% click rate
   drivescore: 32.5% click rate
   improve: 29.9% click rate
   loans: 28.7% click rate
   protect: 28.4% click rate
   insights: 27.7% click rate

   Overall click rate: 30.60%
   Total clicks: 306 out of 1,000 interactions

4. USER ENGAGEMENT FREQUENCY:
   Average interactions per user: 10.0
   Median interactions per user: 9.5
   Most active user: 18 interactions
   Least active user: 2 interactions

5. ADDITIONAL INSIGHTS:
   Average time viewed overall: 29.2 seconds
   Median time viewed: 20.6 seconds
   Longest session: 245.2 seconds

Created 3 user segments:
         total_interactions  avg_time_viewed  click_rate  total_time_viewed  unique_content_viewed
segment                                                                                           
0                     12.92            32.21        0.37             412.26                  11.61
1                      8.37            34.93        0.22             283.47                   7.73
2                      8.35            20.41        0.32             171.44                   7.50

DEBUG: Segment characteristics before naming:
         total_interactions  avg_time_viewed  click_rate  total_time_viewed  unique_content_viewed
segment                                                                                           
0                     12.92            32.21        0.37             412.26                  11.61
1                      8.37            34.93        0.22             283.47                   7.73
2                      8.35            20.41        0.32             171.44                   7.50

DEBUG: Segments ranked by engagement score:
  Rank 1: Segment 0 - Score: 0.569
  Rank 2: Segment 1 - Score: 0.430
  Rank 3: Segment 2 - Score: 0.419

Segment 0:
  Click rate: 0.370
  Avg time viewed: 32.2s
  Total interactions: 12.9
  Engagement Score: 0.569
  -> Assigned: High Engagement (Rank 1)

Segment 1:
  Click rate: 0.220
  Avg time viewed: 34.9s
  Total interactions: 8.4
  Engagement Score: 0.430
  -> Assigned: Medium Engagement (Rank 2)

Segment 2:
  Click rate: 0.320
  Avg time viewed: 20.4s
  Total interactions: 8.3
  Engagement Score: 0.419
  -> Assigned: Low Engagement (Rank 3)
Segment names:  {0: 'High Engagement', 1: 'Medium Engagement', 2: 'Low Engagement'}

Actual user counts per segment:
segment_name
High Engagement      36
Low Engagement       34
Medium Engagement    30


Engagement score statistics:
Segment summary with calculated engagement scores:
  High Engagement: 0.569 score, 36 users
  Medium Engagement: 0.430 score, 30 users
  Low Engagement: 0.419 score, 34 users

============================================================
CONTENT PREFERENCES BY SEGMENT
============================================================
Average content preferences by segment:
                   credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                  
High Engagement           0.193       0.152    0.135     0.178  0.172    0.170
Low Engagement            0.168       0.181    0.203     0.176  0.113    0.158
Medium Engagement         0.134       0.179    0.171     0.184  0.182    0.150

Average time spent on content by segment:
                   viewed  credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                          
High Engagement      32.2          31.8        27.3     23.8      31.6   21.2     28.8
Low Engagement       20.4          17.2        13.7     14.4      14.0   11.0     17.5
Medium Engagement    34.9          23.6        25.5     30.6      21.0   39.0     33.4

Click rates by content type and segment:
                   credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                  
High Engagement           0.462       0.329    0.273     0.245  0.343    0.189
Low Engagement            0.336       0.206    0.353     0.309  0.118    0.137
Medium Engagement         0.117       0.233    0.118     0.133  0.228    0.311

















































============================================================
DAY 1: EXPLORATORY DATA ANALYSIS
============================================================
Dataset shape: (1000, 6)
Date range: 2025-06-09 to 2025-07-09
Number of unique users: 100
Number of unique content pieces: 50

1. MOST COMMON CONTENT TYPES:
   insights: 177 interactions (17.7%)
   improve: 174 interactions (17.4%)
   credit_cards: 167 interactions (16.7%)
   protect: 162 interactions (16.2%)
   drivescore: 160 interactions (16.0%)
   loans: 160 interactions (16.0%)

2. TIME SPENT ON PAGES:
   Average time by content type:
   credit_cards: 30.4s avg, 22.7s median
   drivescore: 27.8s avg, 20.4s median
   improve: 28.5s avg, 16.8s median
   insights: 28.5s avg, 18.1s median
   loans: 26.6s avg, 19.2s median
   protect: 33.3s avg, 24.2s median

3. CLICK RATES BY CONTENT TYPE:
   credit_cards: 36.5% click rate
   drivescore: 32.5% click rate
   improve: 29.9% click rate
   loans: 28.7% click rate
   protect: 28.4% click rate
   insights: 27.7% click rate

   Overall click rate: 30.60%
   Total clicks: 306 out of 1,000 interactions

4. USER ENGAGEMENT FREQUENCY:
   Average interactions per user: 10.0
   Median interactions per user: 9.5
   Most active user: 18 interactions
   Least active user: 2 interactions

5. ADDITIONAL INSIGHTS:
   Average time viewed overall: 29.2 seconds
   Median time viewed: 20.6 seconds
   Longest session: 245.2 seconds



============================================================
DAY 2: USER SEGMENTATION
============================================================
Created 3 user segments:
         total_interactions  avg_time_viewed  click_rate  total_time_viewed  unique_content_viewed
segment                                                                                           
0                     12.92            32.21        0.37             412.26                  11.61
1                      8.37            34.93        0.22             283.47                   7.73
2                      8.35            20.41        0.32             171.44                   7.50

Segment characteristics before naming:
         total_interactions  avg_time_viewed  click_rate  total_time_viewed  unique_content_viewed
segment                                                                                           
0                     12.92            32.21        0.37             412.26                  11.61
1                      8.37            34.93        0.22             283.47                   7.73
2                      8.35            20.41        0.32             171.44                   7.50

Segments ranked by engagement score:
  Rank 1: Segment 0 - Score: 0.569
  Rank 2: Segment 1 - Score: 0.430
  Rank 3: Segment 2 - Score: 0.419

Segment 0:
  Click rate: 0.370
  Avg time viewed: 32.2s
  Total interactions: 12.9
  Engagement Score: 0.569
  -> Assigned: High Engagement (Rank 1)

Segment 1:
  Click rate: 0.220
  Avg time viewed: 34.9s
  Total interactions: 8.4
  Engagement Score: 0.430
  -> Assigned: Medium Engagement (Rank 2)

Segment 2:
  Click rate: 0.320
  Avg time viewed: 20.4s
  Total interactions: 8.3
  Engagement Score: 0.419
  -> Assigned: Low Engagement (Rank 3)
Segment names:  {0: 'High Engagement', 1: 'Medium Engagement', 2: 'Low Engagement'}

Actual user counts per segment:
segment_name
High Engagement      36
Low Engagement       34
Medium Engagement    30
Name: count, dtype: int64

Engagement score statistics:
Segment summary with calculated engagement scores:
  High Engagement: 0.569 score, 36 users
  Medium Engagement: 0.430 score, 30 users
  Low Engagement: 0.419 score, 34 users

============================================================
CONTENT PREFERENCES BY SEGMENT
============================================================
Average content preferences by segment:
                   credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                  
High Engagement           0.193       0.152    0.135     0.178  0.172    0.170
Low Engagement            0.168       0.181    0.203     0.176  0.113    0.158
Medium Engagement         0.134       0.179    0.171     0.184  0.182    0.150

Average time spent on content by segment:
                   viewed  credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                          
High Engagement      32.2          31.8        27.3     23.8      31.6   21.2     28.8
Low Engagement       20.4          17.2        13.7     14.4      14.0   11.0     17.5
Medium Engagement    34.9          23.6        25.5     30.6      21.0   39.0     33.4

Click rates by content type and segment:
                   credit_cards  drivescore  improve  insights  loans  protect
segment_name                                                                  
High Engagement           0.462       0.329    0.273     0.245  0.343    0.189
Low Engagement            0.336       0.206    0.353     0.309  0.118    0.137
Medium Engagement         0.117       0.233    0.118     0.133  0.228    0.311

Financial side
Normalize all financial attributes to be between 0 to 1 and then give each attribute a percentage weighting
Credit card score: 30%
DTI: 25%
Missed payments: 15%
Income: 15%
CCJ: 10%
Assets: 5%


Financial: 
Set thresholds as 33 and 67 percentiles

Leads to changes in recommendations
Stable       34
At_Risk      33
Excellent    33

Segments:
Premium_Engaged      20
Growth_Focused       18
Recovery_Moderate    17
Mainstream           16
Recovery_Engaged     16
Premium_Moderate     13

SEGMENT ANALYSIS BY FINANCIAL HEALTH
============================================================
                  financial_health_score       credit_score            dti_ratio          income           engagement_score
                                    mean   std         mean  min   max      mean   std      mean    median             mean
enhanced_segment                                                                                                           
Growth_Focused                      0.71  0.03       808.56  697   919      0.40  0.15  42529.48  42682.56             0.57
Mainstream                          0.70  0.03       784.31  696   898      0.37  0.16  43516.15  45486.42             0.41
Premium_Engaged                     0.79  0.04       912.55  811  1000      0.26  0.11  45034.34  47152.50             0.58
Premium_Moderate                    0.82  0.05       922.38  810  1000      0.21  0.13  44039.60  43935.58             0.40
Recovery_Engaged                    0.56  0.08       658.75  359   778      0.68  0.23  32577.71  33977.16             0.58
Recovery_Moderate                   0.57  0.07       657.47  419   781      0.64  0.17  36790.07  36175.31             0.42



NEW removing Percentiles and doing absolute threshold:
financial_category
Good         55
Fair         26
Excellent    16
Poor          3
Name: count, dtype: int64
Score ranges: 0.364 - 0.901

Absolute Categorization Thresholds Applied:
Excellent: Score >= 0.8 (Strong financial health across all metrics)
Good: Score >= 0.65 (Above average financial health)
Fair: Score >= 0.45 (Some concerns but manageable)
Poor: Score < 0.45 (Significant financial challenges)

Enhanced segment distribution:
enhanced_segment
Growth_Focused       45
Mainstream           36
Premium_Moderate      9
Premium_Engaged       7
Recovery_Engaged      2
Recovery_Moderate     1
Name: count, dtype: int64
Generating financially-aware recommendations...


================================================================================
ENHANCED FINANCIALLY-AWARE RECOMMENDATIONS
================================================================================

============================================================
SEGMENT ANALYSIS BY FINANCIAL HEALTH
============================================================
                  financial_health_score       credit_score            dti_ratio          income           engagement_score
                                    mean   std         mean  min   max      mean   std      mean    median             mean
enhanced_segment                                                                                                           
Growth_Focused                      0.69  0.08       795.11  529   944      0.45  0.18  40920.80  39373.39             0.57
Mainstream                          0.66  0.08       745.39  519   941      0.49  0.21  39796.95  40970.50             0.42
Premium_Engaged                     0.83  0.02       945.00  900  1000      0.16  0.12  41589.88  42675.06             0.59
Premium_Moderate                    0.85  0.04       940.44  833  1000      0.17  0.14  47296.07  46095.15             0.39
Recovery_Engaged                    0.41  0.06       475.00  359   591      1.09  0.04  27447.66  27447.66             0.63
Recovery_Moderate                   0.41   NaN       419.00  419   419      0.50   NaN  35849.77  35849.77             0.44

Enhanced recommendations saved to 'enhanced_financial_recommendations.csv'


