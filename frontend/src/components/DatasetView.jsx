import { useState } from 'react'
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
import { FileInput } from '@astryxdesign/core/FileInput'
import { Banner } from '@astryxdesign/core/Banner'
import { Divider } from '@astryxdesign/core/Divider'
import { CodeBlock } from '@astryxdesign/core/CodeBlock'
import { Table, pixel, proportional } from '@astryxdesign/core/Table'
import { Badge } from '@astryxdesign/core/Badge'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'

const PIPELINE_SNIPPET = `logit = -1.6 + 0.038*(days - 35)
  + 0.62*(tickets - 1.5)
  - 0.09*(freq - 10)`

const DATASET_COLUMNS = [
  {
    key: 'customer_id',
    header: 'ID',
    width: pixel(110),
    renderCell: (row) => (
      <Text type="code" color="secondary">
        {row.customer_id}
      </Text>
    ),
  },
  {
    key: 'annual_spend',
    header: 'Annual Spend',
    width: proportional(1, { minWidth: 110 }),
    align: 'end',
    renderCell: (row) => (
      <Text hasTabularNumbers>${Number(row.annual_spend).toLocaleString()}</Text>
    ),
  },
  {
    key: 'purchase_frequency',
    header: 'Freq',
    width: pixel(75),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.purchase_frequency}</Text>,
  },
  {
    key: 'avg_cart_value',
    header: 'Avg Cart',
    width: pixel(95),
    align: 'end',
    renderCell: (row) => (
      <Text hasTabularNumbers>${Number(row.avg_cart_value).toFixed(2)}</Text>
    ),
  },
  {
    key: 'days_since_last_visit',
    header: 'Inactive',
    width: pixel(85),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.days_since_last_visit}d</Text>,
  },
  {
    key: 'support_tickets',
    header: 'Tickets',
    width: pixel(80),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.support_tickets}</Text>,
  },
  {
    key: 'tenure_months',
    header: 'Tenure',
    width: pixel(85),
    align: 'end',
    renderCell: (row) => <Text hasTabularNumbers>{row.tenure_months}m</Text>,
  },
  {
    key: 'churned',
    header: 'Churned',
    width: pixel(105),
    renderCell: (row) =>
      row.churned === 1 ? (
        <Badge variant="error" label="Churned" />
      ) : (
        <Text type="supporting" color="secondary">
          Retained
        </Text>
      ),
  },
]

export default function DatasetView({ dataset, onDatasetUpdate }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)

  const handleRegenerate = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/dataset/regenerate', { method: 'POST' })
      if (!res.ok) throw new Error('Failed to regenerate dataset')
      const data = await res.json()
      onDatasetUpdate(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleFileChange = async (files) => {
    const file = Array.isArray(files) ? files[0] : files
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch('/api/dataset/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      onDatasetUpdate(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (!dataset) {
    return (
      <Layout
        height="fill"
        content={
          <LayoutContent padding={6}>
            <Card>
              <Text color="secondary">Loading dataset from FastAPI backend...</Text>
            </Card>
          </LayoutContent>
        }
      />
    )
  }

  return (
    <Layout
      height="fill"
      header={
        <LayoutHeader hasDivider>
          <HStack hAlign="between" vAlign="center" wrap="wrap" gap={4}>
            <VStack gap={0.5}>
              <Heading level={1}>E-Commerce Customer Dataset</Heading>
              <Text type="supporting" color="secondary">
                Member 1 · Shared behavioral dataset powering Segmentation, Churn Prediction, and Dynamic Pricing
              </Text>
            </VStack>
            <Button
              label="Regenerate 1,000 Rows"
              variant="secondary"
              isLoading={loading}
              onClick={handleRegenerate}
            />
          </HStack>
        </LayoutHeader>
      }
      start={
        <LayoutPanel
          hasDivider
          width={320}
          role="complementary"
          label="Dataset Controls and Pipeline Notes"
        >
          <VStack gap={4}>
            <FileInput
              label="Upload Custom CSV"
              accept=".csv"
              mode="dropzone"
              value={selectedFile}
              onChange={setSelectedFile}
              changeAction={handleFileChange}
              isLoading={loading}
              description="CSV must include behavioral features and churned column"
            />

            {error ? (
              <Banner
                status="error"
                title="Dataset error"
                description={error}
                collapsible={false}
              />
            ) : null}

            <Divider />

            <VStack gap={2}>
              <Heading level={4}>Pipeline Note</Heading>
              <CodeBlock
                code={PIPELINE_SNIPPET}
                language="python"
                title="ml/data_pipeline.py"
                width="100%"
                hasCopyButton={false}
              />
              <Text type="supporting" color="secondary">
                Generates 4 customer archetypes with logistic churn probabilities.
              </Text>
            </VStack>
          </VStack>
        </LayoutPanel>
      }
      content={
        <LayoutContent padding={6}>
          <VStack gap={6}>
            <Grid columns={{ minWidth: 200, max: 3, repeat: 'fit' }} gap={4}>
              <Card>
                <VStack gap={1}>
                  <Heading level={4}>Customer Records</Heading>
                  <Heading level={2}>
                    <Text type="inherit" hasTabularNumbers>
                      {dataset.rows.toLocaleString()}
                    </Text>
                  </Heading>
                  <Text type="supporting" color="secondary">
                    Clean behavioral rows
                  </Text>
                </VStack>
              </Card>

              <Card>
                <VStack gap={1}>
                  <Heading level={4}>Historical Churn Rate</Heading>
                  <Heading level={2}>
                    <Text type="inherit" hasTabularNumbers>
                      {dataset.churn_rate}%
                    </Text>
                  </Heading>
                  <Text type="supporting" color="secondary">
                    Target label for Supervised ML
                  </Text>
                </VStack>
              </Card>

              <Card>
                <VStack gap={1}>
                  <Heading level={4}>Avg Annual Spend</Heading>
                  <Heading level={2}>
                    <Text type="inherit" hasTabularNumbers>
                      ${dataset.stats?.annual_spend?.mean?.toLocaleString() ?? '0'}
                    </Text>
                  </Heading>
                  <Text type="supporting" color="secondary" hasTabularNumbers>
                    ${dataset.stats?.annual_spend?.min} – ${dataset.stats?.annual_spend?.max}
                  </Text>
                </VStack>
              </Card>
            </Grid>

            <Card>
              <VStack gap={4}>
                <Heading level={3}>Sample Records (First 15 Rows)</Heading>
                <Table
                  data={dataset.preview}
                  columns={DATASET_COLUMNS}
                  idKey="customer_id"
                  density="compact"
                  dividers="rows"
                  hasHover
                />
              </VStack>
            </Card>
          </VStack>
        </LayoutContent>
      }
    />
  )
}
