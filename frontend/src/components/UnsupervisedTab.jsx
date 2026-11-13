import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import {
  HStack,
  Layout,
  LayoutContent,
  LayoutHeader,
  LayoutPanel,
  VStack,
} from '@astryxdesign/core/Layout'
import { Grid } from '@astryxdesign/core/Grid'
import { Card } from '@astryxdesign/core/Card'
import { Button } from '@astryxdesign/core/Button'
import { Slider } from '@astryxdesign/core/Slider'
import { Divider } from '@astryxdesign/core/Divider'
import { CodeBlock } from '@astryxdesign/core/CodeBlock'
import { Table, pixel, proportional } from '@astryxdesign/core/Table'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'

const CLUSTER_COLORS = [
  'var(--color-data-categorical-blue, #0171E3)',
  'var(--color-data-categorical-green, #0B991F)',
  'var(--color-data-categorical-orange, #EB6E00)',
  'var(--color-data-categorical-purple, #6B1EFD)',
  'var(--color-data-categorical-teal, #008575)',
  'var(--color-data-categorical-pink, #D9186A)',
  'var(--color-data-categorical-yellow, #D19B00)',
  'var(--color-data-neutral, #8494A3)',
]

const UNSUPERVISED_SNIPPET = `X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=k, init='k-means++')
labels = kmeans.fit_predict(X_scaled)`

const PERSONA_COLUMNS = [
  {
    key: 'cluster_id',
    header: 'Cluster',
    width: pixel(85),
    renderCell: (row) => (
      <Text type="code" color="secondary" hasTabularNumbers>
        C{row.cluster_id}
      </Text>
    ),
  },
  {
    key: 'persona',
    header: 'Discovered Persona',
    width: proportional(1, { minWidth: 160 }),
    renderCell: (row) => <Text weight="semibold">{row.persona}</Text>,
  },
  {
    key: 'pct',
    header: 'Share',
    width: pixel(90),
    align: 'end',
    renderCell: (row) => (
      <Text hasTabularNumbers>
        {row.count} ({row.pct}%)
      </Text>
    ),
  },
  {
    key: 'avg_annual_spend',
    header: 'Avg Spend',
    width: pixel(110),
    align: 'end',
    renderCell: (row) => (
      <Text hasTabularNumbers>${row.avg_annual_spend.toLocaleString()}</Text>
    ),
  },
  {
    key: 'avg_purchase_frequency',
    header: 'Freq/yr',
    width: pixel(85),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.avg_purchase_frequency}</Text>,
  },
  {
    key: 'avg_days_since_visit',
    header: 'Inactive',
    width: pixel(85),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.avg_days_since_visit}d</Text>,
  },
  {
    key: 'churn_rate',
    header: 'Churn Rate',
    width: pixel(100),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.churn_rate}%</Text>,
  },
]

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <Card padding={3}>
      <VStack gap={1}>
        {payload.map((entry, idx) => (
          <Text key={idx} type="supporting" hasTabularNumbers>
            {entry.name}: {entry.value}
          </Text>
        ))}
      </VStack>
    </Card>
  )
}

export default function UnsupervisedTab() {
  const [nClusters, setNClusters] = useState(4)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const runClustering = async (k = nClusters) => {
    setLoading(true)
    try {
      const res = await fetch('/api/unsupervised/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_clusters: k }),
      })
      const data = await res.json()
      setResult(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    runClustering(4)
  }, [])

  return (
    <Layout
      height="fill"
      header={
        <LayoutHeader hasDivider>
          <VStack gap={0.5}>
            <Heading level={1}>K-Means Customer Segmentation</Heading>
            <Text type="supporting" color="secondary">
              Member 2 · Unsupervised Learning with StandardScaler, K-Means++, and 2D PCA
            </Text>
          </VStack>
        </LayoutHeader>
      }
      start={
        <LayoutPanel
          hasDivider
          width={320}
          role="complementary"
          label="Clustering Parameters"
        >
          <VStack gap={4}>
            <Slider
              label="Number of Clusters (K)"
              min={2}
              max={8}
              step={1}
              value={nClusters}
              onChange={setNClusters}
              valueDisplay="text"
              formatValue={(v) => `${v} clusters`}
            />

            <Button
              label="Run K-Means"
              variant="primary"
              width="100%"
              isLoading={loading}
              onClick={() => runClustering(nClusters)}
            />

            <Divider />

            <VStack gap={2}>
              <Heading level={4}>Algorithm Note</Heading>
              <CodeBlock
                code={UNSUPERVISED_SNIPPET}
                language="python"
                title="ml/unsupervised.py"
                width="100%"
                hasCopyButton={false}
              />
              <Text type="supporting" color="secondary">
                StandardScaler prevents annual_spend from dominating Euclidean distance.
              </Text>
            </VStack>
          </VStack>
        </LayoutPanel>
      }
      content={
        <LayoutContent padding={6}>
          {result ? (
            <VStack gap={6}>
              <Grid columns={{ minWidth: 200, max: 3, repeat: 'fit' }} gap={4}>
                <Card>
                  <VStack gap={1}>
                    <Heading level={4}>Silhouette Score</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        {result.silhouette_score}
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary">
                      Closer to 1.0 indicates cleaner separation
                    </Text>
                  </VStack>
                </Card>

                <Card>
                  <VStack gap={1}>
                    <Heading level={4}>Inertia (WCSS)</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        {result.inertia.toLocaleString()}
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary">
                      Within-cluster sum of squares
                    </Text>
                  </VStack>
                </Card>

                <Card>
                  <VStack gap={1}>
                    <Heading level={4}>PCA Variance Retained</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        {(result.pca_explained_variance[0] + result.pca_explained_variance[1]).toFixed(1)}%
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary" hasTabularNumbers>
                      PC1: {result.pca_explained_variance[0]}% · PC2: {result.pca_explained_variance[1]}%
                    </Text>
                  </VStack>
                </Card>
              </Grid>

              <Grid columns={{ minWidth: 320, max: 2, repeat: 'fit' }} gap={4}>
                <Card>
                  <VStack gap={3}>
                    <VStack gap={0.5}>
                      <Heading level={3}>2D PCA Cluster Projection</Heading>
                      <Text type="supporting" color="secondary">
                        Customers colored by discovered K-Means segment
                      </Text>
                    </VStack>
                    <ResponsiveContainer width="100%" height={280}>
                      <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, rgba(128,128,128,0.2))" />
                        <XAxis
                          type="number"
                          dataKey="pca_x"
                          name="PC1"
                          tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                        />
                        <YAxis
                          type="number"
                          dataKey="pca_y"
                          name="PC2"
                          tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                        />
                        <ZAxis range={[36, 36]} />
                        <Tooltip content={<ChartTooltip />} />
                        {result.profiles.map((prof) => (
                          <Scatter
                            key={prof.cluster_id}
                            name={prof.persona}
                            data={result.scatter_points.filter((p) => p.cluster === prof.cluster_id)}
                            fill={CLUSTER_COLORS[prof.cluster_id % CLUSTER_COLORS.length]}
                          />
                        ))}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </VStack>
                </Card>

                <Card>
                  <VStack gap={3}>
                    <VStack gap={0.5}>
                      <Heading level={3}>Elbow Method (Inertia vs. K)</Heading>
                      <Text type="supporting" color="secondary">
                        Justifies optimal cluster count (K = 4)
                      </Text>
                    </VStack>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={result.elbow_curve} margin={{ top: 10, right: 16, bottom: 10, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, rgba(128,128,128,0.2))" />
                        <XAxis
                          dataKey="k"
                          tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                        />
                        <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }} />
                        <Tooltip content={<ChartTooltip />} />
                        <Line
                          type="monotone"
                          dataKey="inertia"
                          name="Inertia"
                          stroke="var(--color-data-categorical-green, #0B991F)"
                          strokeWidth={3}
                          dot={{ r: 4, fill: 'var(--color-data-categorical-green, #0B991F)' }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </VStack>
                </Card>
              </Grid>

              <Card>
                <VStack gap={4}>
                  <Heading level={3}>Discovered Customer Personas</Heading>
                  <Table
                    data={result.profiles}
                    columns={PERSONA_COLUMNS}
                    idKey="cluster_id"
                    density="balanced"
                    dividers="rows"
                    hasHover
                  />
                </VStack>
              </Card>
            </VStack>
          ) : (
            <Card>
              <Text color="secondary">Running K-Means segmentation...</Text>
            </Card>
          )}
        </LayoutContent>
      }
    />
  )
}
