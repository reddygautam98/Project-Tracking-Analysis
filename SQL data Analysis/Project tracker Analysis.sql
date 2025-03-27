-- 1. Resource Allocation Analysis
-- Analyze which resource issues are most common and their impact on project timelines
SELECT 
    Reason_Missed,
    COUNT(*) AS Number_Of_Projects,
    ROUND(AVG(Days_Taken), 2) AS Avg_Days_Taken,
    ROUND(AVG(DATEDIFF(Completion_Date, Deadline)), 2) AS Avg_Days_Overdue,
    ROUND(SUM(CASE WHEN Met_Deadline = 'No' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS Missed_Deadline_Percentage
FROM 
    ProjectTracker
GROUP BY 
    Reason_Missed
ORDER BY 
    Number_Of_Projects DESC;

-- 2. Project Health Trends
-- Analyze project health trends over time (by month)
SELECT 
    DATE_FORMAT(Start_Date, '%Y-%m') AS Month,
    COUNT(*) AS Total_Projects,
    SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) AS Projects_On_Time,
    ROUND(SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS On_Time_Percentage,
    ROUND(AVG(Days_Taken), 2) AS Avg_Days_Taken,
    ROUND(AVG(DATEDIFF(Completion_Date, Deadline)), 2) AS Avg_Delay
FROM 
    ProjectTracker
GROUP BY 
    DATE_FORMAT(Start_Date, '%Y-%m')
ORDER BY 
    Month;

-- 3. Dependency Chain Analysis
-- Identify potential dependency chains by analyzing projects that start right after others finish
WITH ProjectTimeline AS (
    SELECT 
        Project_Name,
        Start_Date,
        Completion_Date,
        LEAD(Project_Name) OVER (ORDER BY Start_Date) AS Next_Project,
        LEAD(Start_Date) OVER (ORDER BY Start_Date) AS Next_Project_Start
    FROM 
        ProjectTracker
)
SELECT 
    Project_Name AS Predecessor_Project,
    Next_Project AS Successor_Project,
    Completion_Date AS Predecessor_End,
    Next_Project_Start AS Successor_Start,
    DATEDIFF(Next_Project_Start, Completion_Date) AS Days_Between
FROM 
    ProjectTimeline
WHERE 
    DATEDIFF(Next_Project_Start, Completion_Date) BETWEEN 0 AND 5
ORDER BY 
    Days_Between;

-- 4. Project Complexity Score Analysis
-- Calculate a complexity score based on duration, delays, and reason for missing deadlines
SELECT 
    Project_Name,
    Days_Taken,
    CASE 
        WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline)
        ELSE 0
    END AS Days_Overdue,
    Reason_Missed,
    -- Complexity Score Formula: Days_Taken + (Days_Overdue * 1.5) + Reason_Weight
    Days_Taken + 
    (CASE 
        WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) * 1.5
        ELSE 0
    END) + 
    (CASE 
        WHEN Reason_Missed = 'Technical Problem' THEN 25
        WHEN Reason_Missed = 'Scope Change' THEN 20
        WHEN Reason_Missed = 'Resource Issues' THEN 15
        WHEN Reason_Missed = 'Approval Delay' THEN 10
        WHEN Reason_Missed = 'Delayed' THEN 5
        ELSE 0
    END) AS Complexity_Score
FROM 
    ProjectTracker
ORDER BY 
    Complexity_Score DESC
LIMIT 20;

-- 5. Predictive Delay Analysis
-- Identify patterns that predict delays
SELECT 
    CASE 
        WHEN Days_Taken <= 20 THEN 'Short (<=20 days)'
        WHEN Days_Taken <= 40 THEN 'Medium (21-40 days)'
        ELSE 'Long (>40 days)'
    END AS Project_Duration_Category,
    COUNT(*) AS Total_Projects,
    SUM(CASE WHEN Met_Deadline = 'No' THEN 1 ELSE 0 END) AS Delayed_Projects,
    ROUND(SUM(CASE WHEN Met_Deadline = 'No' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS Delay_Percentage,
    ROUND(AVG(CASE WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) ELSE 0 END), 2) AS Avg_Delay_Days
FROM 
    ProjectTracker
GROUP BY 
    Project_Duration_Category
ORDER BY 
    Delay_Percentage DESC;

-- 6. Complex Project Performance Analysis
-- Analyze performance of complex projects (those with longer durations)
WITH ProjectPerformance AS (
    SELECT 
        Project_Name,
        DATEDIFF(Deadline, Start_Date) AS Planned_Duration,
        Days_Taken AS Actual_Duration,
        (Days_Taken - DATEDIFF(Deadline, Start_Date)) AS Duration_Variance,
        Met_Deadline,
        Reason_Missed
    FROM 
        ProjectTracker
)
SELECT 
    Project_Name,
    Planned_Duration,
    Actual_Duration,
    Duration_Variance,
    ROUND((Duration_Variance / Planned_Duration) * 100, 2) AS Variance_Percentage,
    Met_Deadline,
    Reason_Missed
FROM 
    ProjectPerformance
WHERE 
    Planned_Duration > 30  -- Consider projects with planned duration > 30 days as complex
ORDER BY 
    Variance_Percentage DESC
LIMIT 20;

-- 7. Risk Analysis with Rolling Averages
-- Calculate 3-month rolling averages for project delays
WITH MonthlyMetrics AS (
    SELECT 
        DATE_FORMAT(Start_Date, '%Y-%m') AS Month,
        COUNT(*) AS Total_Projects,
        SUM(CASE WHEN Met_Deadline = 'No' THEN 1 ELSE 0 END) AS Delayed_Projects,
        ROUND(AVG(CASE WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) ELSE 0 END), 2) AS Avg_Delay_Days
    FROM 
        ProjectTracker
    GROUP BY 
        DATE_FORMAT(Start_Date, '%Y-%m')
    ORDER BY 
        Month
)
SELECT 
    m1.Month,
    m1.Total_Projects,
    m1.Delayed_Projects,
    ROUND((m1.Delayed_Projects / m1.Total_Projects) * 100, 2) AS Delay_Percentage,
    m1.Avg_Delay_Days,
    -- Calculate 3-month rolling averages
    ROUND((m1.Delayed_Projects + IFNULL(m2.Delayed_Projects, 0) + IFNULL(m3.Delayed_Projects, 0)) / 
          (m1.Total_Projects + IFNULL(m2.Total_Projects, 0) + IFNULL(m3.Total_Projects, 0)) * 100, 2) AS Rolling_3M_Delay_Percentage,
    ROUND((m1.Avg_Delay_Days + IFNULL(m2.Avg_Delay_Days, 0) + IFNULL(m3.Avg_Delay_Days, 0)) / 
          (CASE WHEN m1.Avg_Delay_Days > 0 THEN 1 ELSE 0 END + 
           CASE WHEN IFNULL(m2.Avg_Delay_Days, 0) > 0 THEN 1 ELSE 0 END + 
           CASE WHEN IFNULL(m3.Avg_Delay_Days, 0) > 0 THEN 1 ELSE 0 END), 2) AS Rolling_3M_Avg_Delay
FROM 
    MonthlyMetrics m1
LEFT JOIN 
    MonthlyMetrics m2 ON m2.Month = DATE_FORMAT(DATE_SUB(STR_TO_DATE(CONCAT(m1.Month, '-01'), '%Y-%m-%d'), INTERVAL 1 MONTH), '%Y-%m')
LEFT JOIN 
    MonthlyMetrics m3 ON m3.Month = DATE_FORMAT(DATE_SUB(STR_TO_DATE(CONCAT(m1.Month, '-01'), '%Y-%m-%d'), INTERVAL 2 MONTH), '%Y-%m')
ORDER BY 
    m1.Month;

-- 8. Resource Utilization and Project Duration Analysis
-- Analyze how different reasons for delays affect project duration
SELECT 
    Reason_Missed,
    COUNT(*) AS Number_Of_Projects,
    ROUND(AVG(Days_Taken), 2) AS Avg_Days_Taken,
    ROUND(MIN(Days_Taken), 2) AS Min_Days_Taken,
    ROUND(MAX(Days_Taken), 2) AS Max_Days_Taken,
    ROUND(STDDEV(Days_Taken), 2) AS StdDev_Days_Taken,
    ROUND(AVG(CASE WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) ELSE 0 END), 2) AS Avg_Delay_Days
FROM 
    ProjectTracker
GROUP BY 
    Reason_Missed
ORDER BY 
    Avg_Days_Taken DESC;

-- 9. Seasonal Trend Analysis
-- Analyze project performance by quarter and year
SELECT 
    YEAR(Start_Date) AS Year,
    QUARTER(Start_Date) AS Quarter,
    COUNT(*) AS Total_Projects,
    SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) AS On_Time_Projects,
    ROUND(SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS On_Time_Percentage,
    ROUND(AVG(Days_Taken), 2) AS Avg_Days_Taken,
    ROUND(AVG(CASE WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) ELSE 0 END), 2) AS Avg_Delay_Days
FROM 
    ProjectTracker
GROUP BY 
    YEAR(Start_Date), QUARTER(Start_Date)
ORDER BY 
    Year, Quarter;

-- 10. Project Health Score Calculation
-- Calculate a health score for each project based on multiple factors
SELECT 
    Project_Name,
    Start_Date,
    Completion_Date,
    Days_Taken,
    Met_Deadline,
    Reason_Missed,
    -- Health Score Components
    CASE WHEN Met_Deadline = 'Yes' THEN 50 ELSE 0 END AS On_Time_Score,
    
    -- Duration Efficiency Score (lower is better, max 25 points)
    CASE 
        WHEN DATEDIFF(Deadline, Start_Date) > 0 THEN
            LEAST(25, 25 * (DATEDIFF(Deadline, Start_Date) / Days_Taken))
        ELSE 0
    END AS Duration_Efficiency_Score,
    
    -- Reason Penalty
    CASE 
        WHEN Reason_Missed = 'Technical Problem' THEN -20
        WHEN Reason_Missed = 'Resource Issues' THEN -15
        WHEN Reason_Missed = 'Scope Change' THEN -10
        WHEN Reason_Missed = 'Approval Delay' THEN -5
        WHEN Reason_Missed = 'Delayed' THEN -10
        ELSE 0
    END AS Reason_Penalty,
    
    -- Delay Severity Penalty (max -25 points)
    CASE 
        WHEN Met_Deadline = 'No' THEN 
            GREATEST(-25, -1 * (DATEDIFF(Completion_Date, Deadline) / 5))
        ELSE 0
    END AS Delay_Severity_Penalty,
    
    -- Calculate Total Health Score (max 100)
    CASE WHEN Met_Deadline = 'Yes' THEN 50 ELSE 0 END +
    CASE 
        WHEN DATEDIFF(Deadline, Start_Date) > 0 THEN
            LEAST(25, 25 * (DATEDIFF(Deadline, Start_Date) / Days_Taken))
        ELSE 0
    END +
    CASE 
        WHEN Reason_Missed = 'Technical Problem' THEN -20
        WHEN Reason_Missed = 'Resource Issues' THEN -15
        WHEN Reason_Missed = 'Scope Change' THEN -10
        WHEN Reason_Missed = 'Approval Delay' THEN -5
        WHEN Reason_Missed = 'Delayed' THEN -10
        ELSE 0
    END +
    CASE 
        WHEN Met_Deadline = 'No' THEN 
            GREATEST(-25, -1 * (DATEDIFF(Completion_Date, Deadline) / 5))
        ELSE 0
    END + 50 AS Health_Score  -- Base score of 50 + components
FROM 
    ProjectTracker
ORDER BY 
    Health_Score DESC;

-- BONUS: Comprehensive Project Dashboard Query
-- Combines multiple metrics into a single dashboard view
SELECT 
    DATE_FORMAT(Start_Date, '%Y-%m') AS Month,
    COUNT(*) AS Total_Projects,
    ROUND(AVG(Days_Taken), 2) AS Avg_Duration,
    SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) AS On_Time_Projects,
    ROUND(SUM(CASE WHEN Met_Deadline = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100, 2) AS On_Time_Percentage,
    ROUND(AVG(CASE WHEN Met_Deadline = 'No' THEN DATEDIFF(Completion_Date, Deadline) ELSE 0 END), 2) AS Avg_Delay_Days,
    -- Most Common Reason for Delay
    (SELECT Reason_Missed 
     FROM ProjectTracker p2 
     WHERE DATE_FORMAT(p2.Start_Date, '%Y-%m') = DATE_FORMAT(p1.Start_Date, '%Y-%m') AND p2.Met_Deadline = 'No'
     GROUP BY Reason_Missed 
     ORDER BY COUNT(*) DESC 
     LIMIT 1) AS Top_Delay_Reason,
    -- Average Health Score
    ROUND(AVG(
        CASE WHEN Met_Deadline = 'Yes' THEN 50 ELSE 0 END +
        CASE 
            WHEN DATEDIFF(Deadline, Start_Date) > 0 THEN
                LEAST(25, 25 * (DATEDIFF(Deadline, Start_Date) / Days_Taken))
            ELSE 0
        END +
        CASE 
            WHEN Reason_Missed = 'Technical Problem' THEN -20
            WHEN Reason_Missed = 'Resource Issues' THEN -15
            WHEN Reason_Missed = 'Scope Change' THEN -10
            WHEN Reason_Missed = 'Approval Delay' THEN -5
            WHEN Reason_Missed = 'Delayed' THEN -10
            ELSE 0
        END +
        CASE 
            WHEN Met_Deadline = 'No' THEN 
                GREATEST(-25, -1 * (DATEDIFF(Completion_Date, Deadline) / 5))
            ELSE 0
        END + 50
    ), 2) AS Avg_Health_Score
FROM 
    ProjectTracker p1
GROUP BY 
    DATE_FORMAT(Start_Date, '%Y-%m')
ORDER BY 
    Month;