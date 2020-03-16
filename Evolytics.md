
# Post Test Analysis for Outdoors Retailer

## A Project for Evolytics by Kelsey Ayers


```python
#Import packages
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
```


```python
df = pd.read_excel("C://Users//klaye//Downloads//Evolytics Data Science Exercise_with key final 10 18 17.xlsx")
```

### Explore the Data


```python
df.shape
```




    (8618, 80)




```python
df.head(5).T.head(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>source_visitor_id</th>
      <td>5271071666666660000000000000000000000</td>
      <td>5274068833333320000000000000000000000</td>
      <td>5280838999999980000000000000000000000</td>
      <td>5281296333333310000000000000000000000</td>
      <td>5281696499999980000000000000000000000</td>
    </tr>
    <tr>
      <th>visit_num</th>
      <td>27</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>First_Visit</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>date</th>
      <td>2016-12-10 00:00:00</td>
      <td>2017-04-12 00:00:00</td>
      <td>2016-12-07 00:00:00</td>
      <td>2016-11-27 00:00:00</td>
      <td>2017-03-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_date</th>
      <td>2016-12-10 00:00:00</td>
      <td>2017-04-12 00:00:00</td>
      <td>2016-12-07 00:00:00</td>
      <td>2016-11-27 00:00:00</td>
      <td>2017-03-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_time</th>
      <td>20:14:14</td>
      <td>16:25:20</td>
      <td>07:01:46</td>
      <td>10:40:43</td>
      <td>15:27:12</td>
    </tr>
    <tr>
      <th>max_timestamp</th>
      <td>2016-12-10 20:17:00</td>
      <td>2017-04-12 16:33:00</td>
      <td>2016-12-07 07:15:00</td>
      <td>2016-11-27 10:58:00</td>
      <td>2017-03-28 15:40:00</td>
    </tr>
    <tr>
      <th>Recipe</th>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
    </tr>
    <tr>
      <th>purchase_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>no_thanks_flag</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>yes_upgrade_flag</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>upgrade_and_purchase</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hit either yes or no</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_product_1</th>
      <td>HIKE-11</td>
      <td>HIKE-16</td>
      <td>HIKE-02</td>
      <td>HIKE-11</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_2</th>
      <td>SOCKS-01</td>
      <td>COOK-08</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_2</th>
      <td>1</td>
      <td>1</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_1</th>
      <td>169.99</td>
      <td>190.99</td>
      <td>158.38</td>
      <td>66.99</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_2</th>
      <td>43.99</td>
      <td>74.99</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>IPD vs NonIPD</th>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
    </tr>
    <tr>
      <th>IPD</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>operating_system_family</th>
      <td>Windows 7</td>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
    </tr>
    <tr>
      <th>browser_family</th>
      <td>AOL</td>
      <td>AppleMail</td>
      <td>AppleMail</td>
      <td>AppleMail</td>
      <td>AppleMail</td>
    </tr>
    <tr>
      <th>user_State</th>
      <td>TX</td>
      <td>NJ</td>
      <td>IL</td>
      <td>CA</td>
      <td>NY</td>
    </tr>
    <tr>
      <th>camp_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hiking__page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>kayak_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>run_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>deals_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>winter_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>snow_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>socks+hiking_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>login_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>review_order_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(5).T.tail(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pageviews_before_popup</th>
      <td>6</td>
      <td>6</td>
      <td>17</td>
      <td>26</td>
      <td>7</td>
    </tr>
    <tr>
      <th>date.1</th>
      <td>2016-12-10 00:00:00</td>
      <td>2017-04-12 00:00:00</td>
      <td>2016-12-07 00:00:00</td>
      <td>2016-11-27 00:00:00</td>
      <td>2017-03-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_date</th>
      <td>2016-12-10 00:00:00</td>
      <td>2017-04-12 00:00:00</td>
      <td>2016-12-07 00:00:00</td>
      <td>2016-11-27 00:00:00</td>
      <td>2017-03-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_time</th>
      <td>20:15:42</td>
      <td>16:29:21</td>
      <td>07:11:46</td>
      <td>10:51:29</td>
      <td>15:34:39</td>
    </tr>
    <tr>
      <th>time_before_popup</th>
      <td>0.001</td>
      <td>0.0028</td>
      <td>0.00694</td>
      <td>0.0075</td>
      <td>0.00517</td>
    </tr>
    <tr>
      <th>hits_before_popup</th>
      <td>13</td>
      <td>20</td>
      <td>84</td>
      <td>77</td>
      <td>52</td>
    </tr>
    <tr>
      <th>OpS_Android</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Chrome_OS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_iOS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Linux</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Mac OS X</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OpS_Ubuntu</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows 10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_8.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_Vista</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_XP</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>AOL_Browser</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AppleMail_Browser</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Chrome_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_iOS_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Edge_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Firefox_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>IE_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mobile_Safari_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Opera_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Other_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Safari_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SeaMonkey_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Test2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>landpage-hiking</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage camping</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage run</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>loyalty_user</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>landpage winter</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SEO</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SEM</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



There are several dates/times present that would be interesting to explore, such as the earliest & latest date, as well as the distribution of upgrades over the course of the test. There are also several (null) values that will need to be handled for in some way. Loyalty user is also showing as True/False rather than 1/0 so let's fix that now.


```python
df['loyalty_user'] = df['loyalty_user'].astype('uint8')
```


```python
df.tail(5).T.head(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>8613</th>
      <th>8614</th>
      <th>8615</th>
      <th>8616</th>
      <th>8617</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>source_visitor_id</th>
      <td>5337262500001280000000000000000000000</td>
      <td>5337270666667950000000000000000000000</td>
      <td>5337621833334630000000000000000000000</td>
      <td>5337940333334640000000000000000000000</td>
      <td>5280487833333320000000000000000000000</td>
    </tr>
    <tr>
      <th>visit_num</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>First_Visit</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>date</th>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-19 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_date</th>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-19 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_time</th>
      <td>07:57:23</td>
      <td>08:30:59</td>
      <td>18:00:42</td>
      <td>16:14:51</td>
      <td>07:46:19</td>
    </tr>
    <tr>
      <th>max_timestamp</th>
      <td>2017-04-18 08:16:00</td>
      <td>2017-04-18 08:55:00</td>
      <td>2017-04-18 18:12:00</td>
      <td>2017-04-19 16:24:00</td>
      <td>2016-11-28 08:05:00</td>
    </tr>
    <tr>
      <th>Recipe</th>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
    </tr>
    <tr>
      <th>purchase_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>no_thanks_flag</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>yes_upgrade_flag</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>upgrade_and_purchase</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hit either yes or no</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_product_1</th>
      <td>COOK-08</td>
      <td>HIKE-11</td>
      <td>HIKE-16</td>
      <td>HIKE-11</td>
      <td>HIKE-02</td>
    </tr>
    <tr>
      <th>purchase_product_2</th>
      <td>HIKE-11</td>
      <td>COOK-08</td>
      <td>(null)</td>
      <td>SOCKS-01</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_3</th>
      <td>RUN-19</td>
      <td>RUN-19</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_units_product_2</th>
      <td>1</td>
      <td>1</td>
      <td>(null)</td>
      <td>1</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_3</th>
      <td>1</td>
      <td>1</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_1</th>
      <td>158.99</td>
      <td>169.99</td>
      <td>295.19</td>
      <td>360.49</td>
      <td>36.99</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_2</th>
      <td>337.98</td>
      <td>74.99</td>
      <td>(null)</td>
      <td>44.09</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_3</th>
      <td>35.99</td>
      <td>35.99</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>IPD vs NonIPD</th>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
    </tr>
    <tr>
      <th>IPD</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>operating_system_family</th>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
      <td>Mac OS X</td>
      <td>Windows 7</td>
    </tr>
    <tr>
      <th>browser_family</th>
      <td>Safari</td>
      <td>Safari</td>
      <td>Safari</td>
      <td>Safari</td>
      <td>SeaMonkey</td>
    </tr>
    <tr>
      <th>user_State</th>
      <td>***</td>
      <td>NJ</td>
      <td>CA</td>
      <td>CA</td>
      <td>TN</td>
    </tr>
    <tr>
      <th>camp_page_flag</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hiking__page_flag</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>kayak_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>run_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>deals_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>winter_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>snow_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>socks+hiking_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>login_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>review_order_flag</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(5).T.tail(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>8613</th>
      <th>8614</th>
      <th>8615</th>
      <th>8616</th>
      <th>8617</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pageviews_before_popup</th>
      <td>13</td>
      <td>6</td>
      <td>7</td>
      <td>13</td>
      <td>5</td>
    </tr>
    <tr>
      <th>date.1</th>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-19 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_date</th>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-18 00:00:00</td>
      <td>2017-04-19 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_time</th>
      <td>08:04:55</td>
      <td>08:43:11</td>
      <td>18:06:22</td>
      <td>16:22:03</td>
      <td>07:54:36</td>
    </tr>
    <tr>
      <th>time_before_popup</th>
      <td>0.0052</td>
      <td>0.0085</td>
      <td>0.0039</td>
      <td>0.005</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>hits_before_popup</th>
      <td>35</td>
      <td>17</td>
      <td>43</td>
      <td>24</td>
      <td>13</td>
    </tr>
    <tr>
      <th>OpS_Android</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Chrome_OS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_iOS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Linux</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Mac OS X</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Ubuntu</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows 10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OpS_Windows8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_8.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_Vista</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_XP</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>AOL_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AppleMail_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_iOS_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Edge_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Firefox_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>IE_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mobile_Safari_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Opera_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Other_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Safari_Browser</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SeaMonkey_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Test2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>landpage-hiking</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage camping</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage run</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>loyalty_user</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage winter</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SEO</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SEM</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


The tail end of the data looks comparable to the head end, but also reveals that the user_state feature has *** values. I will assume those are true NaNs for now, but should confirm with the business users to ensure that this value is not meaningful in some way. Now, let's take a look at the numeric features to get a better sense of their ranges and types.

```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>visit_num</th>
      <td>8618.0</td>
      <td>43.406359</td>
      <td>178.988961</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>3.0000</td>
      <td>11.0000</td>
      <td>3079.0000</td>
    </tr>
    <tr>
      <th>First_Visit</th>
      <td>8618.0</td>
      <td>0.348921</td>
      <td>0.476657</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>purchase_flag</th>
      <td>8618.0</td>
      <td>0.807032</td>
      <td>0.394651</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>no_thanks_flag</th>
      <td>8618.0</td>
      <td>0.815967</td>
      <td>0.387534</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>yes_upgrade_flag</th>
      <td>8618.0</td>
      <td>0.108726</td>
      <td>0.311313</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>upgrade_and_purchase</th>
      <td>8618.0</td>
      <td>0.090160</td>
      <td>0.286428</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>hit either yes or no</th>
      <td>8618.0</td>
      <td>0.924693</td>
      <td>0.277199</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>IPD</th>
      <td>8618.0</td>
      <td>0.047111</td>
      <td>0.211888</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>camp_page_flag</th>
      <td>8618.0</td>
      <td>0.203179</td>
      <td>0.402388</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>hiking__page_flag</th>
      <td>8618.0</td>
      <td>0.202947</td>
      <td>0.402217</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>kayak_page_flag</th>
      <td>8618.0</td>
      <td>0.057206</td>
      <td>0.232249</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>run_page_flag</th>
      <td>8618.0</td>
      <td>0.018102</td>
      <td>0.133327</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>deals_page_flag</th>
      <td>8618.0</td>
      <td>0.005802</td>
      <td>0.075953</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>winter_page_flag</th>
      <td>8618.0</td>
      <td>0.010211</td>
      <td>0.100539</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>snow_page_flag</th>
      <td>8618.0</td>
      <td>0.014388</td>
      <td>0.119093</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>socks+hiking_flag</th>
      <td>8618.0</td>
      <td>0.019610</td>
      <td>0.138664</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>login_page_flag</th>
      <td>8618.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>review_order_flag</th>
      <td>8618.0</td>
      <td>0.998840</td>
      <td>0.034046</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>pageviews_before_popup</th>
      <td>8618.0</td>
      <td>10.411929</td>
      <td>7.756654</td>
      <td>0.000</td>
      <td>6.000</td>
      <td>8.0000</td>
      <td>12.0000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>time_before_popup</th>
      <td>8618.0</td>
      <td>0.005978</td>
      <td>0.006829</td>
      <td>-0.017</td>
      <td>0.002</td>
      <td>0.0036</td>
      <td>0.0071</td>
      <td>0.0944</td>
    </tr>
    <tr>
      <th>hits_before_popup</th>
      <td>8618.0</td>
      <td>28.100719</td>
      <td>22.346373</td>
      <td>0.000</td>
      <td>14.000</td>
      <td>21.0000</td>
      <td>34.0000</td>
      <td>355.0000</td>
    </tr>
    <tr>
      <th>OpS_Android</th>
      <td>8618.0</td>
      <td>0.004990</td>
      <td>0.070464</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Chrome_OS</th>
      <td>8618.0</td>
      <td>0.000580</td>
      <td>0.024081</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_iOS</th>
      <td>8618.0</td>
      <td>0.006962</td>
      <td>0.083153</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Linux</th>
      <td>8618.0</td>
      <td>0.000580</td>
      <td>0.024081</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Mac OS X</th>
      <td>8618.0</td>
      <td>0.127060</td>
      <td>0.333059</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Ubuntu</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows 10</th>
      <td>8618.0</td>
      <td>0.381527</td>
      <td>0.485790</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows_7</th>
      <td>8618.0</td>
      <td>0.399397</td>
      <td>0.489803</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows8</th>
      <td>8618.0</td>
      <td>0.011836</td>
      <td>0.108153</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows_8.1</th>
      <td>8618.0</td>
      <td>0.050940</td>
      <td>0.219888</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows_Vista</th>
      <td>8618.0</td>
      <td>0.008123</td>
      <td>0.089764</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>OpS_Windows_XP</th>
      <td>8618.0</td>
      <td>0.007890</td>
      <td>0.088482</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>8618.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>AOL_Browser</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>AppleMail_Browser</th>
      <td>8618.0</td>
      <td>0.004293</td>
      <td>0.065387</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Chrome_Browser</th>
      <td>8618.0</td>
      <td>0.429102</td>
      <td>0.494977</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_Browser</th>
      <td>8618.0</td>
      <td>0.004293</td>
      <td>0.065387</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_iOS_Browser</th>
      <td>8618.0</td>
      <td>0.000348</td>
      <td>0.018656</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Edge_Browser</th>
      <td>8618.0</td>
      <td>0.075540</td>
      <td>0.264275</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Firefox_Browser</th>
      <td>8618.0</td>
      <td>0.108378</td>
      <td>0.310875</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>IE_Browser</th>
      <td>8618.0</td>
      <td>0.294616</td>
      <td>0.455896</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Mobile_Safari_Browser</th>
      <td>8618.0</td>
      <td>0.006614</td>
      <td>0.081062</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Opera_Browser</th>
      <td>8618.0</td>
      <td>0.000348</td>
      <td>0.018656</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Other_Browser</th>
      <td>8618.0</td>
      <td>0.004409</td>
      <td>0.066260</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Safari_Browser</th>
      <td>8618.0</td>
      <td>0.071826</td>
      <td>0.258215</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>SeaMonkey_Browser</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Test2</th>
      <td>8618.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>landpage-hiking</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>landpage camping</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>landpage run</th>
      <td>8618.0</td>
      <td>0.000116</td>
      <td>0.010772</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>loyalty_user</th>
      <td>8618.0</td>
      <td>0.099211</td>
      <td>0.298962</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>landpage winter</th>
      <td>8618.0</td>
      <td>0.099211</td>
      <td>0.298962</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>SEO</th>
      <td>8618.0</td>
      <td>0.628452</td>
      <td>0.483246</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>SEM</th>
      <td>8618.0</td>
      <td>0.371548</td>
      <td>0.483246</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



There are some interesting revelations here. One notable one is that some users appear to have hit both yes AND no on the upgrade offer popup, indicating that they saw the pop-up twice in a single session. It also appears that there's an issue with the field capturing whether or not users see the login page - I will notes this as an item to review with the client. The time before popup feature also appears to have an anomaly in which there is a negative value. This will need to be explored further. 


```python
df[df['time_before_popup']<0].T.head(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>3767</th>
      <th>4530</th>
      <th>4587</th>
      <th>4588</th>
      <th>4607</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>source_visitor_id</th>
      <td>5334330666667830000000000000000000000</td>
      <td>5275154999999990000000000000000000000</td>
      <td>5280071333333320000000000000000000000</td>
      <td>5280087666666650000000000000000000000</td>
      <td>5281524999999980000000000000000000000</td>
    </tr>
    <tr>
      <th>visit_num</th>
      <td>1</td>
      <td>216</td>
      <td>1007</td>
      <td>943</td>
      <td>187</td>
    </tr>
    <tr>
      <th>First_Visit</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>date</th>
      <td>2017-04-09 00:00:00</td>
      <td>2016-11-26 00:00:00</td>
      <td>2017-04-11 00:00:00</td>
      <td>2017-03-22 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_date</th>
      <td>2017-04-09 00:00:00</td>
      <td>2016-11-26 00:00:00</td>
      <td>2017-04-11 00:00:00</td>
      <td>2017-03-22 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>min_timestamp_time</th>
      <td>16:47:11</td>
      <td>18:54:05</td>
      <td>10:56:59</td>
      <td>03:54:20</td>
      <td>13:49:40</td>
    </tr>
    <tr>
      <th>max_timestamp</th>
      <td>2017-04-09 16:48:00</td>
      <td>2016-11-26 18:54:00</td>
      <td>2017-04-11 10:56:00</td>
      <td>2017-03-22 03:57:00</td>
      <td>2016-11-28 14:38:00</td>
    </tr>
    <tr>
      <th>Recipe</th>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>Recipe B | Upgrade Suggestive</td>
    </tr>
    <tr>
      <th>purchase_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>no_thanks_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>yes_upgrade_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>upgrade_and_purchase</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hit either yes or no</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_product_1</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>HIKE-02</td>
    </tr>
    <tr>
      <th>purchase_product_2</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>COOK-08</td>
    </tr>
    <tr>
      <th>purchase_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_1</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_units_product_2</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>purchase_units_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_units_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_1</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>111.99</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_2</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>30.99</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_3</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>purchase_revenue_product_4</th>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
      <td>(null)</td>
    </tr>
    <tr>
      <th>IPD vs NonIPD</th>
      <td>Non-IPD</td>
      <td>Non-IPD</td>
      <td>IPD</td>
      <td>IPD</td>
      <td>Non-IPD</td>
    </tr>
    <tr>
      <th>IPD</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>operating_system_family</th>
      <td>Android</td>
      <td>Windows 10</td>
      <td>Windows 10</td>
      <td>Windows 10</td>
      <td>Windows 7</td>
    </tr>
    <tr>
      <th>browser_family</th>
      <td>Chrome Mobile</td>
      <td>Firefox</td>
      <td>Firefox</td>
      <td>Firefox</td>
      <td>Firefox</td>
    </tr>
    <tr>
      <th>user_State</th>
      <td>***</td>
      <td>HI</td>
      <td>LA</td>
      <td>LA</td>
      <td>TX</td>
    </tr>
    <tr>
      <th>camp_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>hiking__page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>kayak_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>run_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>deals_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>winter_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>snow_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>socks+hiking_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>login_page_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>review_order_flag</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df['time_before_popup']<0].T.tail(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>3767</th>
      <th>4530</th>
      <th>4587</th>
      <th>4588</th>
      <th>4607</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pageviews_before_popup</th>
      <td>5</td>
      <td>6</td>
      <td>14</td>
      <td>7</td>
      <td>26</td>
    </tr>
    <tr>
      <th>date.1</th>
      <td>2017-04-09 00:00:00</td>
      <td>2016-11-26 00:00:00</td>
      <td>2017-04-11 00:00:00</td>
      <td>2017-03-22 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_date</th>
      <td>2017-04-09 00:00:00</td>
      <td>2016-11-26 00:00:00</td>
      <td>2017-04-11 00:00:00</td>
      <td>2017-03-22 00:00:00</td>
      <td>2016-11-28 00:00:00</td>
    </tr>
    <tr>
      <th>upsell_timestamp_time</th>
      <td>16:33:26</td>
      <td>18:52:51</td>
      <td>10:56:31</td>
      <td>03:48:52</td>
      <td>13:25:13</td>
    </tr>
    <tr>
      <th>time_before_popup</th>
      <td>-0.0095</td>
      <td>-0.0009</td>
      <td>-0.0003</td>
      <td>-0.0038</td>
      <td>-0.017</td>
    </tr>
    <tr>
      <th>hits_before_popup</th>
      <td>10</td>
      <td>11</td>
      <td>29</td>
      <td>24</td>
      <td>81</td>
    </tr>
    <tr>
      <th>OpS_Android</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Chrome_OS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_iOS</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Linux</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Mac OS X</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Ubuntu</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows 10</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>OpS_Windows8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_8.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_Vista</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>OpS_Windows_XP</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Test</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>AOL_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>AppleMail_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_Browser</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chrome_Mobile_iOS_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Edge_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Firefox_Browser</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>IE_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mobile_Safari_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Opera_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Other_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Safari_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SeaMonkey_Browser</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Test2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>landpage-hiking</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage camping</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage run</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>loyalty_user</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>landpage winter</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SEO</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>SEM</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The source of the issue is that the upsell date and time precede the session start date/time (min_timestamp_date/min_timestamp_time). This could potentially be a time zone issue or other system issue. I will drop these from the dataset for now, but will bring this to the client's attention.


```python
df = df[df['time_before_popup']>=0]
```

I also noticed that there are two 'Date' fields in the data. I'd like to verify that these are truly duplicates so I can remove one.


```python
df[df['date']!=df['date.1']].shape
```




    (0, 80)




```python
df = df.drop('date.1',axis=1)
```

Next, should handle the null values I saw in the data ealier.


```python
#Replace Null values previously identified with standard numpy NaN values
df = df.replace('(null)',np.NaN)
df = df.replace('***',np.NaN)
df = df.replace('null',np.NaN)
```


```python
#Fill numeric nulls with zeroes for purchases to help with processing/scaling later on
for col in df.columns:
    if 'revenue' in col or 'units' in col:
        df[col] = df[col].fillna(0)
```


```python
#Ensure my nulls are handled
print(df.isna().sum(axis=0).head(40))
```

    source_visitor_id                0
    visit_num                        0
    First_Visit                      0
    date                             0
    min_timestamp_date               0
    min_timestamp_time               0
    max_timestamp                    0
    Recipe                           0
    purchase_flag                    0
    no_thanks_flag                   0
    yes_upgrade_flag                 0
    upgrade_and_purchase             0
    hit either yes or no             0
    purchase_product_1            1659
    purchase_product_2            6205
    purchase_product_3            8031
    purchase_product_4            8412
    purchase_units_product_1         0
    purchase_units_product_2         0
    purchase_units_product_3         0
    purchase_units_product_4         0
    purchase_revenue_product_1       0
    purchase_revenue_product_2       0
    purchase_revenue_product_3       0
    purchase_revenue_product_4       0
    IPD vs NonIPD                    0
    IPD                              0
    operating_system_family          0
    browser_family                   0
    user_State                     633
    camp_page_flag                   0
    hiking__page_flag                0
    kayak_page_flag                  0
    run_page_flag                    0
    deals_page_flag                  0
    winter_page_flag                 0
    snow_page_flag                   0
    socks+hiking_flag                0
    login_page_flag                  0
    review_order_flag                0
    dtype: int64
    


```python
print(df.isna().sum(axis=0).tail(40))
```

    review_order_flag            0
    pageviews_before_popup       0
    upsell_timestamp_date        5
    upsell_timestamp_time        5
    time_before_popup            0
    hits_before_popup            0
    OpS_Android                  0
    OpS_Chrome_OS                0
    OpS_iOS                      0
    OpS_Linux                    0
    OpS_Mac OS X                 0
    OpS_Ubuntu                   0
    OpS_Windows 10               0
    OpS_Windows_7                0
    OpS_Windows8                 0
    OpS_Windows_8.1              0
    OpS_Windows_Vista            0
    OpS_Windows_XP               0
    Test                         0
    AOL_Browser                  0
    AppleMail_Browser            0
    Chrome_Browser               0
    Chrome_Mobile_Browser        0
    Chrome_Mobile_iOS_Browser    0
    Edge_Browser                 0
    Firefox_Browser              0
    IE_Browser                   0
    Mobile_Safari_Browser        0
    Opera_Browser                0
    Other_Browser                0
    Safari_Browser               0
    SeaMonkey_Browser            0
    Test2                        0
    landpage-hiking              0
    landpage camping             0
    landpage run                 0
    loyalty_user                 0
    landpage winter              0
    SEO                          0
    SEM                          0
    dtype: int64
    

It's logical to see nulls in purchase products since not all sessions resulted in a purchase and not all customers who made a purchase did so for more than one product. I am going to leave those as is. User State is also missing values, but these cannot be reasonably imputed. The blanks themselves could be meaningful, so I am going to leave these too. We have nulls in the upgrade date/timestamp, though, which is unexpected since these users were part of the test and thus should have been prompted to upgrade. Let's explore further.


```python
df[df['upsell_timestamp_date'].isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_visitor_id</th>
      <th>visit_num</th>
      <th>First_Visit</th>
      <th>date</th>
      <th>min_timestamp_date</th>
      <th>min_timestamp_time</th>
      <th>max_timestamp</th>
      <th>Recipe</th>
      <th>purchase_flag</th>
      <th>no_thanks_flag</th>
      <th>...</th>
      <th>Safari_Browser</th>
      <th>SeaMonkey_Browser</th>
      <th>Test2</th>
      <th>landpage-hiking</th>
      <th>landpage camping</th>
      <th>landpage run</th>
      <th>loyalty_user</th>
      <th>landpage winter</th>
      <th>SEO</th>
      <th>SEM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3652</th>
      <td>5336396833334580000000000000000000000</td>
      <td>1</td>
      <td>1</td>
      <td>2017-04-15</td>
      <td>2017-04-15</td>
      <td>00:00:12</td>
      <td>2017-04-15 00:06:00</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4130</th>
      <td>5303289166666610000000000000000000000</td>
      <td>7</td>
      <td>0</td>
      <td>2017-03-23</td>
      <td>2017-03-23</td>
      <td>23:59:56</td>
      <td>2017-03-23 23:59:00</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5108</th>
      <td>5318699666667170000000000000000000000</td>
      <td>2</td>
      <td>0</td>
      <td>2017-04-12</td>
      <td>2017-04-12</td>
      <td>23:51:24</td>
      <td>2017-04-12 23:52:00</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5146</th>
      <td>5321231333333940000000000000000000000</td>
      <td>3</td>
      <td>0</td>
      <td>2017-03-29</td>
      <td>2017-03-29</td>
      <td>23:43:13</td>
      <td>2017-03-29 23:58:00</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5914</th>
      <td>5278005166666650000000000000000000000</td>
      <td>1</td>
      <td>1</td>
      <td>2016-12-12</td>
      <td>2016-12-12</td>
      <td>00:00:09</td>
      <td>2016-12-12 00:01:00</td>
      <td>Recipe B | Upgrade Suggestive</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>



Since these observations fall outside of the bounds of the test, I will drop these. It is also noteworthy that these visits all hover around midnight. This would be something that would be worthwhile for the client to correct.


```python
df = df[df['upsell_timestamp_date'].notnull()]
```

### Feature Engineering

Now I'm going to drop some features that aren't meaningful and create new ones to deepen the analysis.


```python
#Drop features not needed for this analysis
for col in df.columns:
    if 'Recipe' in col or 'Test' in col or 'login' in col:
        df = df.drop(col,axis=1)
```


```python
#Compute total revenue and units ordered for each visit
df['Total Revenue']=df['purchase_revenue_product_1']+df['purchase_revenue_product_2']+df['purchase_revenue_product_3']+df['purchase_revenue_product_4']
df['Purchases per Order']=df['purchase_units_product_1']+df['purchase_units_product_2']+df['purchase_units_product_3']+df['purchase_units_product_4']

#Parse out date
df['Month'] = df['date'].dt.strftime("%Y-%m")
df['Weekday'] = df['date'].dt.strftime("%A")

#Bucket number of visits
visit_buckets = [0,1,5,10,25,100,4000]
df['visit_buckets'] = pd.cut(df['visit_num'],visit_buckets).astype(str)

#Handle for cases where the landing page of a user was none of the 4 represented in the dataset
df['landpage other'] = np.where(df['landpage-hiking']+df['landpage camping']+
                                df['landpage run']+df['landpage winter']==0, 1, 0)
```

### Data Preparation


```python
#Clean up and categorize my variables. Convert categorical variables to dummies.
#End result will be two datasets - one with dummy variables for correlation analysis, the other with categorical names for easier visualization
df_viz = pd.DataFrame()
df_numeric = pd.DataFrame()
df_dummies = pd.DataFrame()

for col in df.columns[1:]:
	attName = col
	dType = df[attName].dtype
	if 'date' in attName or 'time' in attName:
		df_viz = pd.concat([df_viz, df[attName]],axis=1)
	elif dType == object:
		df_dummies = pd.concat([df_dummies, pd.get_dummies(df[col],prefix=col)], axis=1)
		df_viz = pd.concat([df_viz, df[attName]],axis=1)
	elif df[attName].max(axis=0)==1 and df[attName].min(axis=0)==0:
		df_dummies = pd.concat([df_dummies, df[col]], axis=1)
		df_viz = pd.concat([df_viz, df[attName]], axis=1)
	else:
		df_numeric = pd.concat([df_numeric, df[attName]], axis=1)
```


```python
#Use sklearn to normalize the numeric features
numeric_names = df_numeric.columns
df_numeric = pd.DataFrame(preprocessing.scale(df_numeric),columns=numeric_names)

df_viz=pd.concat([df_numeric,df_viz], axis=1)
df_dummies = pd.concat([df_numeric,df_dummies], axis=1)
```

### Basic Data Exploration


```python
sns.distplot(df['Total Revenue'])
plt.axvline(df['Total Revenue'].mean(), color='r', linestyle='--')
print(df['Total Revenue'].mean())
```

    162.66181459107054
    


![png](output_39_1.png)



```python
sns.distplot(df['Purchases per Order'])
plt.axvline(df['Purchases per Order'].mean(), color='r', linestyle='--')
print(df['Purchases per Order'].mean())
```

    1.1785548327137547
    


![png](output_40_1.png)


### Evaluate how to maintain revenue & orders in the B Test

I'd like to do a basic correlation analysis to see the relationship between the features in the dataset and Revenue. I'll start off with looking at the features that have a negative relationship with revenue to see if any of these would have been a result of the B test.


```python
corr_revenue = df_dummies.corr().round(decimals=4).sort_values(by=['Total Revenue'],ascending=False)

for col in corr_revenue.columns:
    if 'Revenue' in col or 'revenue' in col or 'units' in col or 'user_State' in col or 'purchase_product' in col or 'Browser' in col or 'browser' in col or 'OpS' in col or 'operating' in col or 'HIKE' in col:
        corr_revenue = corr_revenue.drop(col,axis=0)

corr_revenue[['Total Revenue']].style.background_gradient(cmap='coolwarm')
```




<style  type="text/css" >
    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow0_col0 {
            background-color:  #b40426;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow1_col0 {
            background-color:  #e2dad5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow2_col0 {
            background-color:  #cad8ef;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow3_col0 {
            background-color:  #84a7fc;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow4_col0 {
            background-color:  #7597f6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow5_col0 {
            background-color:  #7396f5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow6_col0 {
            background-color:  #6f92f3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow7_col0 {
            background-color:  #6b8df0;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow8_col0 {
            background-color:  #6788ee;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow9_col0 {
            background-color:  #6687ed;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow10_col0 {
            background-color:  #6687ed;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow11_col0 {
            background-color:  #6485ec;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow12_col0 {
            background-color:  #6180e9;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow13_col0 {
            background-color:  #5e7de7;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow14_col0 {
            background-color:  #5d7ce6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow15_col0 {
            background-color:  #5d7ce6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow16_col0 {
            background-color:  #5d7ce6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow17_col0 {
            background-color:  #5b7ae5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow18_col0 {
            background-color:  #5b7ae5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow19_col0 {
            background-color:  #5a78e4;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow20_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow21_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow22_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow23_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow24_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow25_col0 {
            background-color:  #5977e3;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow26_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow27_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow28_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow29_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow30_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow31_col0 {
            background-color:  #5875e1;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow32_col0 {
            background-color:  #5673e0;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow33_col0 {
            background-color:  #5572df;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow34_col0 {
            background-color:  #5572df;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow35_col0 {
            background-color:  #5572df;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow36_col0 {
            background-color:  #5470de;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow37_col0 {
            background-color:  #5470de;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow38_col0 {
            background-color:  #5470de;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow39_col0 {
            background-color:  #5470de;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow40_col0 {
            background-color:  #516ddb;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow41_col0 {
            background-color:  #516ddb;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow42_col0 {
            background-color:  #4e68d8;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow43_col0 {
            background-color:  #4c66d6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow44_col0 {
            background-color:  #4c66d6;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow45_col0 {
            background-color:  #4b64d5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow46_col0 {
            background-color:  #4b64d5;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow47_col0 {
            background-color:  #4961d2;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow48_col0 {
            background-color:  #465ecf;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow49_col0 {
            background-color:  #4358cb;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow50_col0 {
            background-color:  #4257c9;
        }    #T_9de95afa_674a_11ea_ac1b_5cea1d92559crow51_col0 {
            background-color:  #3b4cc0;
        }</style>  
<table id="T_9de95afa_674a_11ea_ac1b_5cea1d92559c" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Total Revenue</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row0" class="row_heading level0 row0" >Purchases per Order</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow0_col0" class="data row0 col0" >0.6963</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row1" class="row_heading level0 row1" >hit either yes or no</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow1_col0" class="data row1 col0" >0.3271</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row2" class="row_heading level0 row2" >purchase_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow2_col0" class="data row2 col0" >0.258</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row3" class="row_heading level0 row3" >no_thanks_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow3_col0" class="data row3 col0" >0.0989</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row4" class="row_heading level0 row4" >Month_2016-12</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow4_col0" class="data row4 col0" >0.064</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row5" class="row_heading level0 row5" >upgrade_and_purchase</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow5_col0" class="data row5 col0" >0.0628</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row6" class="row_heading level0 row6" >Month_2016-11</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow6_col0" class="data row6 col0" >0.0525</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row7" class="row_heading level0 row7" >snow_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow7_col0" class="data row7 col0" >0.0428</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row8" class="row_heading level0 row8" >winter_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow8_col0" class="data row8 col0" >0.0336</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row9" class="row_heading level0 row9" >IPD</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow9_col0" class="data row9 col0" >0.0313</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row10" class="row_heading level0 row10" >IPD vs NonIPD_IPD</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow10_col0" class="data row10 col0" >0.0313</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row11" class="row_heading level0 row11" >yes_upgrade_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow11_col0" class="data row11 col0" >0.0274</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row12" class="row_heading level0 row12" >visit_buckets_(100, 4000]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow12_col0" class="data row12 col0" >0.0207</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row13" class="row_heading level0 row13" >socks+hiking_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow13_col0" class="data row13 col0" >0.0152</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row14" class="row_heading level0 row14" >SEO</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow14_col0" class="data row14 col0" >0.0097</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row15" class="row_heading level0 row15" >visit_buckets_(0, 1]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow15_col0" class="data row15 col0" >0.0094</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row16" class="row_heading level0 row16" >First_Visit</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow16_col0" class="data row16 col0" >0.0094</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row17" class="row_heading level0 row17" >Weekday_Tuesday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow17_col0" class="data row17 col0" >0.0087</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row18" class="row_heading level0 row18" >visit_num</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow18_col0" class="data row18 col0" >0.0072</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row19" class="row_heading level0 row19" >Weekday_Monday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow19_col0" class="data row19 col0" >0.0049</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row20" class="row_heading level0 row20" >Weekday_Friday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow20_col0" class="data row20 col0" >0.0029</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row21" class="row_heading level0 row21" >Weekday_Thursday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow21_col0" class="data row21 col0" >0.0024</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row22" class="row_heading level0 row22" >pageviews_before_popup</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow22_col0" class="data row22 col0" >0.0023</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row23" class="row_heading level0 row23" >landpage camping</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow23_col0" class="data row23 col0" >0.0005</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row24" class="row_heading level0 row24" >landpage other</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow24_col0" class="data row24 col0" >0.0003</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row25" class="row_heading level0 row25" >review_order_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow25_col0" class="data row25 col0" >0.0002</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row26" class="row_heading level0 row26" >landpage winter</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow26_col0" class="data row26 col0" >0</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row27" class="row_heading level0 row27" >loyalty_user</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow27_col0" class="data row27 col0" >0</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row28" class="row_heading level0 row28" >visit_buckets_(25, 100]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow28_col0" class="data row28 col0" >-0.0001</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row29" class="row_heading level0 row29" >visit_buckets_(1, 5]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow29_col0" class="data row29 col0" >-0.0002</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row30" class="row_heading level0 row30" >Weekday_Wednesday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow30_col0" class="data row30 col0" >-0.0012</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row31" class="row_heading level0 row31" >Month_2017-06</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow31_col0" class="data row31 col0" >-0.0025</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row32" class="row_heading level0 row32" >landpage run</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow32_col0" class="data row32 col0" >-0.0032</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row33" class="row_heading level0 row33" >visit_buckets_(10, 25]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow33_col0" class="data row33 col0" >-0.007</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row34" class="row_heading level0 row34" >landpage-hiking</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow34_col0" class="data row34 col0" >-0.0072</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row35" class="row_heading level0 row35" >Weekday_Sunday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow35_col0" class="data row35 col0" >-0.0078</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row36" class="row_heading level0 row36" >deals_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow36_col0" class="data row36 col0" >-0.009</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row37" class="row_heading level0 row37" >SEM</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow37_col0" class="data row37 col0" >-0.0097</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row38" class="row_heading level0 row38" >Month_2017-02</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow38_col0" class="data row38 col0" >-0.0103</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row39" class="row_heading level0 row39" >Month_2017-01</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow39_col0" class="data row39 col0" >-0.0115</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row40" class="row_heading level0 row40" >Month_2017-07</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow40_col0" class="data row40 col0" >-0.0156</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row41" class="row_heading level0 row41" >Month_2017-05</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow41_col0" class="data row41 col0" >-0.0156</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row42" class="row_heading level0 row42" >visit_buckets_(5, 10]</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow42_col0" class="data row42 col0" >-0.0266</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row43" class="row_heading level0 row43" >run_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow43_col0" class="data row43 col0" >-0.0273</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row44" class="row_heading level0 row44" >Weekday_Saturday</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow44_col0" class="data row44 col0" >-0.0283</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row45" class="row_heading level0 row45" >hiking__page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow45_col0" class="data row45 col0" >-0.0308</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row46" class="row_heading level0 row46" >IPD vs NonIPD_Non-IPD</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow46_col0" class="data row46 col0" >-0.0313</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row47" class="row_heading level0 row47" >Month_2017-03</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow47_col0" class="data row47 col0" >-0.0382</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row48" class="row_heading level0 row48" >kayak_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow48_col0" class="data row48 col0" >-0.0443</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row49" class="row_heading level0 row49" >hits_before_popup</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow49_col0" class="data row49 col0" >-0.0518</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row50" class="row_heading level0 row50" >camp_page_flag</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow50_col0" class="data row50 col0" >-0.0562</td> 
    </tr>    <tr> 
        <th id="T_9de95afa_674a_11ea_ac1b_5cea1d92559clevel0_row51" class="row_heading level0 row51" >Month_2017-04</th> 
        <td id="T_9de95afa_674a_11ea_ac1b_5cea1d92559crow51_col0" class="data row51 col0" >-0.0752</td> 
    </tr></tbody> 
</table> 



It's interesting that the winter page, snow page, IPD and socks+hiking had positive correlations with revenue. It's likewise interesting that certain page views and that Saturday orders are negatively correlated. I'll explore those further in Tableau. Before doing that, I'll also explore the hit either yes or no feature to see why that is so strongly correlated.

Let's see if there are any other interesting insights in a correlation analysis for Purchases per Order.


```python
corr_orders = df_dummies.corr().round(decimals=4).sort_values(by=['Purchases per Order'],ascending=False)

for col in corr_orders.columns:
    if 'Revenue' in col or 'revenue' in col or 'units' in col or 'user_State' in col or 'purchase_product' in col or 'Browser' in col or 'browser' in col or 'OpS' in col or 'operating' in col or 'HIKE' in col:
        corr_orders = corr_orders.drop(col,axis=0)

corr_orders[['Purchases per Order']].style.background_gradient(cmap='coolwarm')
```




<style  type="text/css" >
    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow0_col0 {
            background-color:  #b40426;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow1_col0 {
            background-color:  #cbd8ee;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow2_col0 {
            background-color:  #b5cdfa;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow3_col0 {
            background-color:  #7699f6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow4_col0 {
            background-color:  #688aef;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow5_col0 {
            background-color:  #6687ed;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow6_col0 {
            background-color:  #6687ed;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow7_col0 {
            background-color:  #6180e9;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow8_col0 {
            background-color:  #5d7ce6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow9_col0 {
            background-color:  #5d7ce6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow10_col0 {
            background-color:  #5a78e4;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow11_col0 {
            background-color:  #5977e3;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow12_col0 {
            background-color:  #5875e1;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow13_col0 {
            background-color:  #5673e0;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow14_col0 {
            background-color:  #5572df;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow15_col0 {
            background-color:  #5572df;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow16_col0 {
            background-color:  #5572df;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow17_col0 {
            background-color:  #5470de;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow18_col0 {
            background-color:  #5470de;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow19_col0 {
            background-color:  #5470de;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow20_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow21_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow22_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow23_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow24_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow25_col0 {
            background-color:  #536edd;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow26_col0 {
            background-color:  #516ddb;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow27_col0 {
            background-color:  #506bda;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow28_col0 {
            background-color:  #506bda;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow29_col0 {
            background-color:  #506bda;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow30_col0 {
            background-color:  #506bda;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow31_col0 {
            background-color:  #506bda;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow32_col0 {
            background-color:  #4f69d9;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow33_col0 {
            background-color:  #4f69d9;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow34_col0 {
            background-color:  #4f69d9;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow35_col0 {
            background-color:  #4e68d8;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow36_col0 {
            background-color:  #4e68d8;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow37_col0 {
            background-color:  #4e68d8;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow38_col0 {
            background-color:  #4e68d8;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow39_col0 {
            background-color:  #4e68d8;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow40_col0 {
            background-color:  #4c66d6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow41_col0 {
            background-color:  #4c66d6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow42_col0 {
            background-color:  #4c66d6;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow43_col0 {
            background-color:  #4b64d5;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow44_col0 {
            background-color:  #4b64d5;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow45_col0 {
            background-color:  #4a63d3;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow46_col0 {
            background-color:  #4a63d3;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow47_col0 {
            background-color:  #4a63d3;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow48_col0 {
            background-color:  #4a63d3;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow49_col0 {
            background-color:  #445acc;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow50_col0 {
            background-color:  #4358cb;
        }    #T_7f081c8c_6755_11ea_8df2_5cea1d92559crow51_col0 {
            background-color:  #3b4cc0;
        }</style>  
<table id="T_7f081c8c_6755_11ea_8df2_5cea1d92559c" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Purchases per Order</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row0" class="row_heading level0 row0" >Purchases per Order</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow0_col0" class="data row0 col0" >1</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row1" class="row_heading level0 row1" >hit either yes or no</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow1_col0" class="data row1 col0" >0.3912</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row2" class="row_heading level0 row2" >purchase_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow2_col0" class="data row2 col0" >0.3135</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row3" class="row_heading level0 row3" >no_thanks_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow3_col0" class="data row3 col0" >0.1206</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row4" class="row_heading level0 row4" >upgrade_and_purchase</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow4_col0" class="data row4 col0" >0.0772</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row5" class="row_heading level0 row5" >Month_2016-12</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow5_col0" class="data row5 col0" >0.0724</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row6" class="row_heading level0 row6" >snow_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow6_col0" class="data row6 col0" >0.0693</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row7" class="row_heading level0 row7" >Month_2016-11</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow7_col0" class="data row7 col0" >0.0535</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row8" class="row_heading level0 row8" >pageviews_before_popup</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow8_col0" class="data row8 col0" >0.0418</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row9" class="row_heading level0 row9" >winter_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow9_col0" class="data row9 col0" >0.0388</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row10" class="row_heading level0 row10" >yes_upgrade_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow10_col0" class="data row10 col0" >0.0322</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row11" class="row_heading level0 row11" >socks+hiking_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow11_col0" class="data row11 col0" >0.0277</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row12" class="row_heading level0 row12" >visit_buckets_(10, 25]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow12_col0" class="data row12 col0" >0.0221</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row13" class="row_heading level0 row13" >visit_buckets_(100, 4000]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow13_col0" class="data row13 col0" >0.018</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row14" class="row_heading level0 row14" >Weekday_Sunday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow14_col0" class="data row14 col0" >0.015</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row15" class="row_heading level0 row15" >IPD</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow15_col0" class="data row15 col0" >0.0138</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row16" class="row_heading level0 row16" >IPD vs NonIPD_IPD</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow16_col0" class="data row16 col0" >0.0138</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row17" class="row_heading level0 row17" >visit_num</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow17_col0" class="data row17 col0" >0.0121</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row18" class="row_heading level0 row18" >Month_2017-06</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow18_col0" class="data row18 col0" >0.0105</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row19" class="row_heading level0 row19" >Weekday_Tuesday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow19_col0" class="data row19 col0" >0.0104</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row20" class="row_heading level0 row20" >deals_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow20_col0" class="data row20 col0" >0.0089</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row21" class="row_heading level0 row21" >loyalty_user</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow21_col0" class="data row21 col0" >0.0088</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row22" class="row_heading level0 row22" >landpage winter</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow22_col0" class="data row22 col0" >0.0088</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row23" class="row_heading level0 row23" >SEO</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow23_col0" class="data row23 col0" >0.0085</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row24" class="row_heading level0 row24" >visit_buckets_(25, 100]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow24_col0" class="data row24 col0" >0.0055</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row25" class="row_heading level0 row25" >Weekday_Monday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow25_col0" class="data row25 col0" >0.0055</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row26" class="row_heading level0 row26" >review_order_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow26_col0" class="data row26 col0" >0.0044</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row27" class="row_heading level0 row27" >hits_before_popup</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow27_col0" class="data row27 col0" >0.0004</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row28" class="row_heading level0 row28" >Weekday_Thursday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow28_col0" class="data row28 col0" >0</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row29" class="row_heading level0 row29" >landpage camping</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow29_col0" class="data row29 col0" >-0.0022</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row30" class="row_heading level0 row30" >landpage run</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow30_col0" class="data row30 col0" >-0.0022</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row31" class="row_heading level0 row31" >landpage-hiking</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow31_col0" class="data row31 col0" >-0.0022</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row32" class="row_heading level0 row32" >visit_buckets_(0, 1]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow32_col0" class="data row32 col0" >-0.004</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row33" class="row_heading level0 row33" >First_Visit</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow33_col0" class="data row33 col0" >-0.004</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row34" class="row_heading level0 row34" >Weekday_Friday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow34_col0" class="data row34 col0" >-0.0069</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row35" class="row_heading level0 row35" >visit_buckets_(1, 5]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow35_col0" class="data row35 col0" >-0.0082</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row36" class="row_heading level0 row36" >SEM</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow36_col0" class="data row36 col0" >-0.0085</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row37" class="row_heading level0 row37" >landpage other</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow37_col0" class="data row37 col0" >-0.0086</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row38" class="row_heading level0 row38" >Weekday_Wednesday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow38_col0" class="data row38 col0" >-0.0094</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row39" class="row_heading level0 row39" >Month_2017-02</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow39_col0" class="data row39 col0" >-0.0106</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row40" class="row_heading level0 row40" >Month_2017-07</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow40_col0" class="data row40 col0" >-0.0119</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row41" class="row_heading level0 row41" >Month_2017-01</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow41_col0" class="data row41 col0" >-0.013</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row42" class="row_heading level0 row42" >IPD vs NonIPD_Non-IPD</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow42_col0" class="data row42 col0" >-0.0138</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row43" class="row_heading level0 row43" >Weekday_Saturday</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow43_col0" class="data row43 col0" >-0.0187</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row44" class="row_heading level0 row44" >run_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow44_col0" class="data row44 col0" >-0.0198</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row45" class="row_heading level0 row45" >camp_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow45_col0" class="data row45 col0" >-0.0204</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row46" class="row_heading level0 row46" >hiking__page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow46_col0" class="data row46 col0" >-0.022</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row47" class="row_heading level0 row47" >Month_2017-05</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow47_col0" class="data row47 col0" >-0.0226</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row48" class="row_heading level0 row48" >visit_buckets_(5, 10]</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow48_col0" class="data row48 col0" >-0.0234</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row49" class="row_heading level0 row49" >Month_2017-03</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow49_col0" class="data row49 col0" >-0.0442</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row50" class="row_heading level0 row50" >kayak_page_flag</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow50_col0" class="data row50 col0" >-0.047</td> 
    </tr>    <tr> 
        <th id="T_7f081c8c_6755_11ea_8df2_5cea1d92559clevel0_row51" class="row_heading level0 row51" >Month_2017-04</th> 
        <td id="T_7f081c8c_6755_11ea_8df2_5cea1d92559crow51_col0" class="data row51 col0" >-0.0793</td> 
    </tr></tbody> 
</table> 



There are no substantive differences in what drives orders versus revenue, so I'm going to move on to visual data exploration.


```python
chart_1 = sns.countplot(x='hit either yes or no',data=df)
```


![png](output_47_0.png)


Unfortunately the positive correlation between variables is because nobody who exited the popup purchased anything and because those that hit either yes or no are highly represented in the dataset. At this point, my hypothesis remains that users deleted items from their basket post-upgrade or otherwise limited their purchases. A way to overcome this might be to recommend additional products for purchase or offer a small discount on certain products if users upgrade, incentivizing them to continue purchasing. I'd like to see which products would be good contenders for such a promotion.


```python
#Add an id to every column
df['id']=df.index

#Restructure the data to make the products easier to report on
df_products = df.melt(id_vars='id',value_vars=['upgrade_and_purchase','purchase_product_1','purchase_product_2','purchase_product_3','purchase_product_4',
                  'purchase_revenue_product_1','purchase_revenue_product_2','purchase_revenue_product_3','purchase_revenue_product_4'])
```


```python
#Create one dataframe for revenue and another for products so that they can be joined back together
df_revenue = df_products[df_products['variable'].str.contains('revenue')==True]
df_revenue.rename(columns={'value':'Revenue'},inplace=True)
df_revenue.reset_index(inplace=True)
df_revenue = pd.DataFrame(df_revenue['Revenue'])

df_upgpur = df_products[df_products['variable']=='upgrade_and_purchase']
df_upgpur.rename(columns={'value':'Upgrade'},inplace=True)
df_upgpur.reset_index(inplace=True)
df_upgpur = pd.DataFrame(df_upgpur['Upgrade'])

df_products = df_products[df_products['variable'].str.contains('revenue')==False]
df_products = df_products[df_products['variable']!='upgrade_and_purchase']
df_products.rename(columns={'value':'Product'},inplace=True)
df_products = pd.DataFrame(df_products[['id','Product']])
df_products.reset_index(inplace=True)
```

    C:\Users\klaye\Anaconda3\lib\site-packages\pandas\core\frame.py:3781: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return super(DataFrame, self).rename(**kwargs)
    


```python
df_products = pd.concat([df_products,df_revenue,df_upgpur],axis=1)
```


```python
#We're already offering a hiking product upgrade, so I'd like to focus on non-hiking products
df_products = df_products[df_products['Product'].notna()]
```


```python
search = []
for values in df_products['Product']:
    search.append(re.search('\w+', values).group())

df_products['Category'] = search
```


```python
#See the 5 the most popular products users purchased
chart_1 = sns.countplot(y='Product',data=df_products,order=df_products.Product.value_counts().iloc[:5].index)
```


![png](output_54_0.png)



```python
#See the top 5 products by their average revenue
chart_2 = sns.barplot(y='Product',x='Revenue',data=df_products,order=df_products.Product.value_counts().iloc[:5].index,ci=None)
```


![png](output_55_0.png)


Based on this, I would recommend offering a small discount on COOK-08. It's a popular product, so there is double incentive for users to upgrade their hiking product AND purchase the COOK-08 product. It also has high average revenue, so it would help drive our revenue KPI higher. Also, by incentivizing users to purchase more, the number of orders per customer should also increase.


```python
#Export curated datasets to Excel for use in Tableau
df.to_excel("C://Users//klaye//Documents//OnlineRetailer-BTest.xlsx")
df_products.to_excel("C://Users//klaye//Documents//OnlineRetailer-Products.xlsx")
```
