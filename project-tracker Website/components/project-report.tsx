"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from "@/components/ui/chart"
import { generateChartData } from "@/lib/chart-data"

export default function ProjectReport() {
  const [chartData, setChartData] = useState<any>(null)

  useEffect(() => {
    // Generate chart data when component mounts
    const data = generateChartData()
    setChartData(data)
  }, [])

  if (!chartData) {
    return <div className="flex items-center justify-center min-h-screen">Loading charts...</div>
  }

  const COLORS = {
    primary: "#3498db",
    secondary: "#2ecc71",
    tertiary: "#9b59b6",
    warning: "#f39c12",
    danger: "#e74c3c",
    dark: "#34495e",
    light: "#ecf0f1",
    success: "#2ecc71",
    info: "#3498db",
  }

  const PIE_COLORS = [COLORS.success, COLORS.danger]
  const BAR_COLORS = [COLORS.primary, COLORS.tertiary, COLORS.warning, COLORS.danger, COLORS.secondary]

  return (
    <div className="max-w-7xl mx-auto p-4 md:p-8">
      <div className="bg-[#34495e] text-white p-6 rounded-lg mb-8 text-center">
        <h1 className="text-3xl font-bold mb-2">Project Tracking Analysis Report</h1>
        <p>Generated on {new Date().toLocaleString()}</p>
      </div>

      <Tabs defaultValue="summary" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="summary">Summary</TabsTrigger>
          <TabsTrigger value="eda">Exploratory Analysis</TabsTrigger>
          <TabsTrigger value="modeling">Predictive Modeling</TabsTrigger>
          <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
          <TabsTrigger value="forecasting">Forecasting</TabsTrigger>
        </TabsList>

        {/* Executive Summary Tab */}
        <TabsContent value="summary">
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Total Projects</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">500</div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Deadline Compliance</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">46.0%</div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Average Delay</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">3.6 days</div>
                </div>
              </div>

              <p className="mb-4">
                This report provides a comprehensive analysis of project tracking data, including exploratory analysis,
                predictive modeling, anomaly detection, and forecasting. Key insights and recommendations are provided
                to improve project management practices and outcomes.
              </p>

              <div className="bg-red-100 text-red-800 p-4 rounded-md mb-6">
                <strong>Key Finding:</strong> Critical: Low deadline compliance rate indicates significant project
                management issues.
              </div>

              <h2 className="text-xl font-bold mb-4">Recommendations</h2>
              <ol className="list-decimal pl-5 space-y-2">
                <li>
                  Implement a comprehensive review of project management processes to address the critical issue of
                  missed deadlines.
                </li>
                <li>Address the most common reason for delays ('Scope Change') with targeted interventions.</li>
                <li>
                  Prepare for an increasing trend in project completions by ensuring adequate resources are available.
                </li>
                <li>
                  Investigate anomalous projects identified in the analysis to understand root causes and prevent
                  similar issues.
                </li>
                <li>
                  Implement a real-time monitoring system to track project progress and identify potential issues early.
                </li>
              </ol>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Exploratory Data Analysis Tab */}
        <TabsContent value="eda">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardContent className="pt-6">
                <h3 className="text-xl font-bold mb-4">Project Deadline Compliance</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={chartData.deadlineCompliance}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                      >
                        {chartData.deadlineCompliance.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-xl font-bold mb-4">Distribution of Project Delays</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData.delayDistribution} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill={COLORS.tertiary} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-xl font-bold mb-4">Top Reasons for Missed Deadlines</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={chartData.missedReasons}
                      margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="name" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill={COLORS.warning}>
                        {chartData.missedReasons.map((entry: any, index: number) => (
                          <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-6">
                <h3 className="text-xl font-bold mb-4">Project Duration Analysis</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid />
                      <XAxis type="number" dataKey="planned" name="Planned Duration" unit=" days" />
                      <YAxis type="number" dataKey="actual" name="Actual Duration" unit=" days" />
                      <ZAxis type="number" dataKey="delay" name="Delay" unit=" days" />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <Legend />
                      <Scatter name="Projects" data={chartData.durationAnalysis} fill={COLORS.primary} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Predictive Modeling Tab */}
        <TabsContent value="modeling">
          <Card>
            <CardContent className="pt-6">
              <h3 className="text-xl font-bold mb-4">Model Performance Comparison</h3>
              <div className="h-80 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData.modelPerformance} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[75, 90]} label={{ value: "Accuracy (%)", angle: -90, position: "insideLeft" }} />
                    <Tooltip formatter={(value) => [`${value}%`, "Accuracy"]} />
                    <Legend />
                    <Bar dataKey="accuracy" fill={COLORS.secondary}>
                      {chartData.modelPerformance.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <h3 className="text-xl font-bold mb-4">Feature Importance</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={chartData.featureImportance}
                    margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="importance" fill={COLORS.info}>
                      {chartData.featureImportance.map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Anomaly Detection Tab */}
        <TabsContent value="anomalies">
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Anomalous Projects</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">50</div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Percentage</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">10.0%</div>
                </div>
              </div>

              <h3 className="text-xl font-bold mb-4">Anomalous Projects</h3>
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid />
                    <XAxis type="number" dataKey="planned" name="Planned Duration" unit=" days" />
                    <YAxis type="number" dataKey="actual" name="Actual Duration" unit=" days" />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Legend />
                    <Scatter name="Normal Projects" data={chartData.anomalyDetection.normal} fill={COLORS.primary} />
                    <Scatter
                      name="Anomalous Projects"
                      data={chartData.anomalyDetection.anomalies}
                      fill={COLORS.danger}
                      shape="star"
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Forecasting Tab */}
        <TabsContent value="forecasting">
          <Card>
            <CardContent className="pt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Daily Avg (Next 30 Days)</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">1.61</div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Total (Next 30 Days)</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">1096.0</div>
                </div>
                <div className="bg-white rounded-lg p-4 shadow text-center">
                  <h3 className="text-lg font-medium text-gray-700">Trend</h3>
                  <div className="text-2xl font-bold text-primary mt-2 mb-2">Increasing</div>
                </div>
              </div>

              <h3 className="text-xl font-bold mb-4">Project Completions Forecast</h3>
              <div className="h-80 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.forecast} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      stroke={COLORS.primary}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      name="Actual"
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      stroke={COLORS.secondary}
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="Forecast"
                    />
                    <Line
                      type="monotone"
                      dataKey="upper"
                      stroke={COLORS.tertiary}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      name="Upper Bound"
                    />
                    <Line
                      type="monotone"
                      dataKey="lower"
                      stroke={COLORS.tertiary}
                      strokeWidth={1}
                      strokeDasharray="3 3"
                      name="Lower Bound"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <h3 className="text-xl font-bold mb-4">Forecast Components</h3>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.forecastComponents} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="trend" stroke={COLORS.primary} strokeWidth={2} name="Trend" />
                    <Line type="monotone" dataKey="seasonal" stroke={COLORS.warning} strokeWidth={2} name="Seasonal" />
                    <Line
                      type="monotone"
                      dataKey="weekly"
                      stroke={COLORS.tertiary}
                      strokeWidth={2}
                      name="Weekly Pattern"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <div className="text-center mt-8 pt-6 border-t text-gray-500 text-sm">
        <p>Generated by Enhanced MLTracker - Advanced Project Tracking & Analysis System</p>
        <p>{new Date().toLocaleString()}</p>
      </div>
    </div>
  )
}

