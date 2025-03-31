export function generateChartData() {
  // Deadline compliance data
  const deadlineCompliance = [
    { name: "Met", value: 46 },
    { name: "Missed", value: 54 },
  ]

  // Delay distribution data
  const delayDistribution = [
    { name: "-5 to 0", count: 230 },
    { name: "1 to 5", count: 120 },
    { name: "6 to 10", count: 80 },
    { name: "11 to 15", count: 40 },
    { name: "16+", count: 30 },
  ]

  // Missed reasons data
  const missedReasons = [
    { name: "Scope Change", count: 85 },
    { name: "Resource Shortage", count: 65 },
    { name: "Technical Issues", count: 55 },
    { name: "External Factors", count: 40 },
    { name: "Poor Planning", count: 35 },
  ]

  // Duration analysis data
  const durationAnalysis = Array(50)
    .fill(0)
    .map((_, i) => {
      const planned = 10 + Math.floor(Math.random() * 90)
      const delay = Math.floor(Math.random() * 20) - 5
      return {
        planned,
        actual: planned + delay,
        delay,
      }
    })

  // Model performance data
  const modelPerformance = [
    { name: "LOGISTIC", accuracy: 85.33 },
    { name: "RF", accuracy: 84.0 },
    { name: "XGB", accuracy: 82.67 },
    { name: "LGB", accuracy: 82.67 },
    { name: "CATBOOST", accuracy: 84.67 },
    { name: "STACKING", accuracy: 86.0 },
  ]

  // Feature importance data
  const featureImportance = [
    { name: "Planned Duration", importance: 0.35 },
    { name: "Project Complexity", importance: 0.25 },
    { name: "Start Quarter", importance: 0.15 },
    { name: "Deadline Extended", importance: 0.12 },
    { name: "Start Month", importance: 0.08 },
    { name: "Start Day of Week", importance: 0.05 },
  ]

  // Anomaly detection data
  const anomalyDetection = {
    normal: Array(40)
      .fill(0)
      .map((_, i) => {
        const planned = 10 + Math.floor(Math.random() * 90)
        const delay = Math.floor(Math.random() * 15) - 5
        return {
          planned,
          actual: planned + delay,
          id: i,
        }
      }),
    anomalies: Array(10)
      .fill(0)
      .map((_, i) => {
        const planned = 10 + Math.floor(Math.random() * 90)
        // Anomalies have much larger deviations
        const delay = (Math.random() > 0.5 ? 1 : -1) * (30 + Math.floor(Math.random() * 40))
        return {
          planned,
          actual: Math.max(5, planned + delay),
          id: i + 100,
        }
      }),
  }

  // Forecast data
  const forecast = []
  const now = new Date()

  // Past data (actual)
  for (let i = 0; i < 12; i++) {
    const date = new Date(now)
    date.setMonth(now.getMonth() - 12 + i)

    const baseValue = 50 + Math.sin(i * 0.5) * 10
    const randomVariation = Math.random() * 15 - 7.5

    forecast.push({
      date: date.toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
      actual: Math.round(baseValue + randomVariation),
      forecast: null,
      upper: null,
      lower: null,
    })
  }

  // Future data (forecast)
  const lastActualValue = forecast[forecast.length - 1].actual

  for (let i = 0; i < 12; i++) {
    const date = new Date(now)
    date.setMonth(now.getMonth() + i)

    const trend = i * 1.5
    const seasonal = Math.sin((i + 12) * 0.5) * 10
    const forecastValue = Math.round(lastActualValue + trend + seasonal)

    forecast.push({
      date: date.toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
      actual: null,
      forecast: forecastValue,
      upper: forecastValue + Math.round(10 + i * 0.5),
      lower: Math.max(0, forecastValue - Math.round(10 + i * 0.5)),
    })
  }

  // Forecast components data
  const forecastComponents = []

  for (let i = 0; i < 24; i++) {
    const date = new Date(now)
    date.setMonth(now.getMonth() - 12 + i)

    forecastComponents.push({
      date: date.toLocaleDateString("en-US", { month: "short", year: "2-digit" }),
      trend: 40 + i * 1.5,
      seasonal: Math.sin(i * 0.5) * 10,
      weekly: Math.cos(i * 2) * 5,
    })
  }

  return {
    deadlineCompliance,
    delayDistribution,
    missedReasons,
    durationAnalysis,
    modelPerformance,
    featureImportance,
    anomalyDetection,
    forecast,
    forecastComponents,
  }
}

