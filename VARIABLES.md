# Variable Descriptions

The lagged variables are aggregated using the specified function from the table over 3-month, 6-month, and 12-month periods. The variables in the tables represent the original monthly time series. Notably, even if the original monthly variable was binary, the lagged variable can be continuous, depending on the aggregation method (e.g., sum). If the "Aggregate Function" column contains a "-", it indicates that the original variable was used directly without any lags.

## Basis Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| SEX | Sex of the member (1- female, 0- male) | binary | - |
| AGE | Age in years on reference date | continuous | - |
| BIRTHMONTH | Birthmonth of member | continuous | - |
| ADVERT_personal, ADVERT_union, ADVERT_none | Was there advertising (personal or from union) involved in the person joining? | binary | - |
| ONLINEENTRY | Did the person sign up for the union online? | binary | - |
| MEMBERSHIP_LENGTH | Membership length in years | continuous | - |
| BANK_i | Is the member registered by bank i on reference date? *(Note: included as some banks correlate with churns)* | binary | - |
| POSTCODE_current | Member's current postcode | continuous | - |
| `EXIT` | Does the member churn in the next nine months after reference date? | binary | - |

## Benefits Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| BAMOUNT_i | Amount of money received by benefit i | continuous | sum |
| BAMOUNT_TOTAL | Total amount of money received from different benefits | continuous | sum |
| BAMOUNT_BINARY | Did the member receive money from any benefits? | binary | max |

## Seminar Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| SEMINARTYPE_i | Member participated in seminar i | binary | max |
| SEMINAR_TOTAL | Total number of seminars the member participated in | continuous | sum |

## Strike Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| STRIKE_LEN | Number of strike days | continuous | sum |
| STRIKE_MONEY | Strike compensation money | continuous | sum |

## Documents Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| TEMPLATE_i | Template document i was send by the union to the member, like "welcome letter" *(Note: excluded "membership termination confirmation" for data leakage reasons)* | binary | max |

## Firm Data

These variables always refer to the member's company.

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| firm_ACTION_i | Union organised action i in the company | binary | max |
| firm_CONTACTPERSON | Is there a union's contact person at the company? | binary | max |
| firm_REPRESENTATIVE_ratio | Ratio of union representatives to number of union members at company | continuous | mean |
| firm_SUPPORT_REPRESENTATIVE_ratio | Ratio of union representative for confidential support to number of union members at company | continuous | mean |
| firm_COMPANYTYPE_i | Type of the company, e.g. group or holding | binary | max |
| firm_EMPLOYEES_ratio | Ratio of employees to union members at the company | continuous | mean |
| firm_EMPLOYEES_F, firm_EMPLOYEES_M | Ratio of female / male employees to union members at the company | continuous | mean |
| firm_SECTOR_i | Sector of the company, e.g. healthcare, municipalities, or education | binary | max |
| firm_REPRESENTATIVE_WrC, firm_REPRESENTATIVE_StC, firm_REPRESENTATIVE_WrCStC | Is there a union's represantative in the company's works council, staff council, or both? | binary | max |
| firm_YOUTHRATIO | Percentage of young people at the company | continuous | mean |
| firm_MEMBERS_TOTAL | Total number of union members at the company | continuous | mean |
| firm_MEMBERS_F, firm_MEMBERS_M | Percentage of female / male union members at the company | continuous | mean |
| firm_MEMBERS_FULLTIME, firm_MEMBERS_PARTTIME | Percentage of full-time / part-time union members at the company | continuous | mean |
| firm_VALID_TARIFF | Is there a valid tariff at the company? | binary | max |

## Committee Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| COMMROLE_i_j | One-hot categories for all commitees i and roles j that appear for more than 1% of the data. Corresponds to internal roles like "Chief Executive Officer" and committees like "Works Council" or "In-house bargaining committee". COMMROLE_999999 for combination with appearance less than 1% | binary | max |
| RANK_max | Maximum rank of all committee and role combinations of member (using responsibility ranking of combinations) | continuous | max |
| RANK_sum | Sum of ranks of all committee and role combinations of member (using responsibility ranking of combinations) | continuous | sum |

## Contact Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| IN_CONTACT | Has the member contacted the union? | binary | sum |
| CATEGORY_i | One-hot categories for all contact reasons, e.g. labour legal protection advice or update of contact details | binary | max |
| STATUS_SENTIMENT | Mean sentiment of the contact (1- positive, 0- neutral, -1- negative), e.g. cancellation by the member has -1, in progress has 0, and issue resolved has 1 | continuous | mean |

## Dynamic Data

| **Variable** | **Description** | **Data Type** | **Aggregate Function** |
| --- | --- | --- | --- |
| POSTCODE_change | Did the postcode change compared to last month? | binary | sum |
| FIRM_change | Did the member change the company since last month? | binary | sum |
| CONTRIBUTION | The member's contribution for the union | continuous | mean |
| CONTRIBUTION_exemption | Was there an exemption regulation for the member fee? | binary | max |
| CONTRIBUTION_CYCLE_i | Member's payment rhythm is i (Y- yearly, H- half yearly, Q- quarterly,  M- monthly) | binary | max |
| CONTRIBUTION_IN_INSTALLMENTS | Is the membership fee paid in installments? | binary | max |
| PAYDELAY_i | Status i of delay in member fee payment | binary | max |
| EMPLOYMENT_DEGREE | Member's percentage of employment with 100% being full-time | continuous | mean |
| EMPLOYMENT_STATUS_i | Member's status of employment, e.g. student, parental leave, or retired | binary | max |
| EMPLOYMENT_TYPE_i | Member's employment type, e.g. civil servant or self-employed | binary | max |
| INCOME | Member's income | continuous | mean |
| TARIFFNEGO | Was there an average monthly contribution increase in the member's tariff category? | binary | max |
| UNDER_TARIFF | Is the member on a tariff plan? | binary | max |
| MAGAZINE | Does the member receive the union's magazine? | binary | max |