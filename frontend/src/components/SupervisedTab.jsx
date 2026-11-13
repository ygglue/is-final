import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
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
import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core/SegmentedControl'
import { Banner } from '@astryxdesign/core/Banner'
import { Divider } from '@astryxdesign/core/Divider'
import { CodeBlock } from '@astryxdesign/core/CodeBlock'
import { Table, pixel, proportional } from '@astryxdesign/core/Table'
import { Badge } from '@astryxdesign/core/Badge'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'

const SUPERVISED_SNIPPET = `clf = RandomForestClassifier(
  n_estimators=n_estimators,
  max_depth=max_depth
)
clf.fit(X_train, y_train)`

const CONFUSION_COLUMNS = [
  {
    key: 'outcome',
    header: 'Classification Outcome',
    width: proportional(1, { minWidth: 160 }),
    renderCell: (row) => <Text weight="medium">{row.outcome}</Text>,
  },
  {
    key: 'meaning',
    header: 'Business Meaning',
    width: proportional(1, { minWidth: 140 }),
    renderCell: (row) => (
      <Text type="supporting" color="secondary">
        {row.meaning}
      </Text>
    ),
  },
  {
    key: 'count',
    header: 'Count',
    width: pixel(85),
    align: 'end',
    renderCell: (row) => (
      <Text weight="bold" hasTabularNumbers>
        {row.count}
      </Text>
    ),
  },
  {
    key: 'status',
    header: 'Result',
    width: pixel(95),
    renderCell: (row) =>
      row.isError ? (
        <Badge variant="error" label="Missed" />
      ) : (
        <Text type="supporting" color="secondary">
          {row.tag}
        </Text>
      ),
  },
]

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  return (
    <Card padding={3}>
      <VStack gap={1}>
        {payload.map((entry, idx) => (
          <Text key={idx} type="supporting" hasTabularNumbers>
            {entry.payload?.feature ?? entry.name}: {entry.value}%
          </Text>
        ))}
      </VStack>
    </Card>
  )
}

export default function SupervisedTab() {
  const [algorithm, setAlgorithm] = useState('random_forest')
  const [nEstimators, setNEstimators] = useState(100)
  const [maxDepth, setMaxDepth] = useState(8)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)

  const [customer, setCustomer] = useState({
    annual_spend: 900,
    purchase_frequency: 10,
    avg_cart_value: 90,
    days_since_last_visit: 45,
    support_tickets: 3,
    tenure_months: 14,
  })
  const [prediction, setPrediction] = useState(null)

  const trainModel = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/supervised/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          algorithm,
          n_estimators: nEstimators,
          max_depth: maxDepth,
        }),
      })
      const data = await res.json()
      setMetrics(data)
    } finally {
      setLoading(false)
    }
  }

  const predictCustomer = async (profile = customer) => {
    const res = await fetch('/api/supervised/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(profile),
    })
    const data = await res.json()
    setPrediction(data)
  }

  useEffect(() => {
    trainModel()
    predictCustomer()
  }, [])

  const updateCustomerField = (field, val) => {
    setCustomer({ ...customer, [field]: Number(val) })
  }

  const confusionRows = metrics
    ? [
        {
          id: 'tn',
          outcome: 'True Negative',
          meaning: 'Retained customer predicted correctly',
          count: metrics.confusion_matrix.tn,
          tag: 'Correct',
          isError: false,
        },
        {
          id: 'tp',
          outcome: 'True Positive',
          meaning: 'Churned customer caught early',
          count: metrics.confusion_matrix.tp,
          tag: 'Correct',
          isError: false,
        },
        {
          id: 'fp',
          outcome: 'False Positive',
          meaning: 'False alarm retention offer',
          count: metrics.confusion_matrix.fp,
          tag: 'Warning',
          isError: false,
        },
        {
          id: 'fn',
          outcome: 'False Negative',
          meaning: 'Churned customer missed',
          count: metrics.confusion_matrix.fn,
          tag: 'Missed',
          isError: true,
        },
      ]
    : []

  return (
    <Layout
      height="fill"
      header={
        <LayoutHeader hasDivider>
          <VStack gap={0.5}>
            <Heading level={1}>Customer Churn Classification</Heading>
            <Text type="supporting" color="secondary">
              Member 3 · Supervised Learning on labeled historical outcomes (80/20 stratified split)
            </Text>
          </VStack>
        </LayoutHeader>
      }
      start={
        <LayoutPanel
          hasDivider
          width={320}
          role="complementary"
          label="Classifier Parameters"
        >
          <VStack gap={4}>
            <VStack gap={1.5}>
              <Text type="label">Algorithm</Text>
              <SegmentedControl
                label="Classifier Algorithm"
                value={algorithm}
                onChange={setAlgorithm}
                layout="fill"
              >
                <SegmentedControlItem value="random_forest" label="Random Forest" />
                <SegmentedControlItem value="logistic_regression" label="Logistic Reg" />
              </SegmentedControl>
            </VStack>

            {algorithm === 'random_forest' ? (
              <VStack gap={3}>
                <Slider
                  label="Trees (n_estimators)"
                  min={20}
                  max={200}
                  step={10}
                  value={nEstimators}
                  onChange={setNEstimators}
                  valueDisplay="text"
                  formatValue={(v) => `${v} trees`}
                />
                <Slider
                  label="Max Depth"
                  min={2}
                  max={16}
                  step={1}
                  value={maxDepth}
                  onChange={setMaxDepth}
                  valueDisplay="text"
                  formatValue={(v) => `depth ${v}`}
                />
              </VStack>
            ) : null}

            <Button
              label="Train & Evaluate"
              variant="primary"
              width="100%"
              isLoading={loading}
              onClick={trainModel}
            />

            <Divider />

            <VStack gap={2}>
              <Heading level={4}>Algorithm Note</Heading>
              <CodeBlock
                code={SUPERVISED_SNIPPET}
                language="python"
                title="ml/supervised.py"
                width="100%"
                hasCopyButton={false}
              />
              <Text type="supporting" color="secondary">
                Bootstrap bagging aggregates trees to reduce variance and overfitting.
              </Text>
            </VStack>
          </VStack>
        </LayoutPanel>
      }
      content={
        <LayoutContent padding={6}>
          <VStack gap={6}>
            {metrics ? (
              <>
                <Grid columns={{ minWidth: 180, max: 4, repeat: 'fit' }} gap={4}>
                  {[
                    { label: 'Test Accuracy', value: `${metrics.accuracy}%` },
                    { label: 'Precision', value: `${metrics.precision}%` },
                    { label: 'Recall', value: `${metrics.recall}%` },
                    { label: 'F1-Score', value: `${metrics.f1_score}%` },
                  ].map((item) => (
                    <Card key={item.label}>
                      <VStack gap={1}>
                        <Heading level={4}>{item.label}</Heading>
                        <Heading level={2}>
                          <Text type="inherit" hasTabularNumbers>
                            {item.value}
                          </Text>
                        </Heading>
                        <Text type="supporting" color="secondary" hasTabularNumbers>
                          {metrics.test_samples} unseen test rows
                        </Text>
                      </VStack>
                    </Card>
                  ))}
                </Grid>

                <Grid columns={{ minWidth: 320, max: 2, repeat: 'fit' }} gap={4}>
                  <Card>
                    <VStack gap={3}>
                      <VStack gap={0.5}>
                        <Heading level={3}>Feature Importance (%)</Heading>
                        <Text type="supporting" color="secondary">
                          Top behavioral drivers of customer churn
                        </Text>
                      </VStack>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          data={metrics.feature_importance}
                          layout="vertical"
                          margin={{ top: 5, right: 20, left: 40, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, rgba(128,128,128,0.2))" />
                          <XAxis
                            type="number"
                            tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                          />
                          <YAxis
                            type="category"
                            dataKey="feature"
                            width={130}
                            tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                          />
                          <Tooltip content={<ChartTooltip />} />
                          <Bar
                            dataKey="importance"
                            fill="var(--color-data-categorical-blue, #0171E3)"
                            radius={[0, 6, 6, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </VStack>
                  </Card>

                  <Card>
                    <VStack gap={3}>
                      <VStack gap={0.5}>
                        <Heading level={3}>Confusion Matrix</Heading>
                        <Text type="supporting" color="secondary">
                          Actual vs. Predicted across {metrics.test_samples} customers
                        </Text>
                      </VStack>
                      <Table
                        data={confusionRows}
                        columns={CONFUSION_COLUMNS}
                        idKey="id"
                        density="compact"
                        dividers="rows"
                        hasHover
                      />
                    </VStack>
                  </Card>
                </Grid>
              </>
            ) : (
              <Card>
                <Text color="secondary">Training supervised classifier...</Text>
              </Card>
            )}

            <Card>
              <VStack gap={4}>
                <HStack hAlign="between" vAlign="center" wrap="wrap" gap={3}>
                  <VStack gap={0.5}>
                    <Heading level={3}>Live Customer Churn Simulator</Heading>
                    <Text type="supporting" color="secondary">
                      Test real-time model inference with hypothetical profiles
                    </Text>
                  </VStack>
                  <Button
                    label="Predict Churn Risk"
                    variant="secondary"
                    onClick={() => predictCustomer()}
                  />
                </HStack>

                <Grid columns={{ minWidth: 220, max: 3, repeat: 'fit' }} gap={4}>
                  <Slider
                    label="Annual Spend ($)"
                    min={50}
                    max={6000}
                    step={50}
                    value={customer.annual_spend}
                    onChange={(v) => updateCustomerField('annual_spend', v)}
                    valueDisplay="text"
                    formatValue={(v) => `$${v}`}
                  />
                  <Slider
                    label="Purchase Frequency (/yr)"
                    min={1}
                    max={45}
                    step={1}
                    value={customer.purchase_frequency}
                    onChange={(v) => updateCustomerField('purchase_frequency', v)}
                    valueDisplay="text"
                    formatValue={(v) => `${v}/yr`}
                  />
                  <Slider
                    label="Avg Cart Value ($)"
                    min={15}
                    max={400}
                    step={5}
                    value={customer.avg_cart_value}
                    onChange={(v) => updateCustomerField('avg_cart_value', v)}
                    valueDisplay="text"
                    formatValue={(v) => `$${v}`}
                  />
                  <Slider
                    label="Days Since Last Visit"
                    min={1}
                    max={150}
                    step={1}
                    value={customer.days_since_last_visit}
                    onChange={(v) => updateCustomerField('days_since_last_visit', v)}
                    valueDisplay="text"
                    formatValue={(v) => `${v}d`}
                  />
                  <Slider
                    label="Support Tickets"
                    min={0}
                    max={10}
                    step={1}
                    value={customer.support_tickets}
                    onChange={(v) => updateCustomerField('support_tickets', v)}
                    valueDisplay="text"
                    formatValue={(v) => `${v} tickets`}
                  />
                  <Slider
                    label="Tenure (Months)"
                    min={1}
                    max={60}
                    step={1}
                    value={customer.tenure_months}
                    onChange={(v) => updateCustomerField('tenure_months', v)}
                    valueDisplay="text"
                    formatValue={(v) => `${v} mo`}
                  />
                </Grid>

                {prediction ? (
                  <Banner
                    status={prediction.predicted_class === 1 ? 'error' : 'success'}
                    title={`${prediction.label} — ${prediction.churn_probability}% Churn Probability (${prediction.risk_tier})`}
                    description={`Evaluated with ${prediction.algorithm}`}
                    collapsible={false}
                  />
                ) : null}
              </VStack>
            </Card>
          </VStack>
        </LayoutContent>
      }
    />
  )
}
