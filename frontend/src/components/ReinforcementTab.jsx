import { useEffect, useState } from 'react'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
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
import { Badge } from '@astryxdesign/core/Badge'
import { Heading } from '@astryxdesign/core/Heading'
import { Text } from '@astryxdesign/core/Text'

const BELLMAN_SNIPPET = `best_next = np.max(q_table[next_state])
td_target = reward + gamma * best_next
q_table[s, a] += alpha * (td_target - q_table[s, a])`

const ACTION_BADGE_VARIANTS = ['neutral', 'cyan', 'orange', 'pink']

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <Card padding={3}>
      <VStack gap={1}>
        <Text type="supporting" color="secondary" hasTabularNumbers>
          Episode {label}
        </Text>
        {payload.map((entry, idx) => (
          <Text key={idx} type="supporting" hasTabularNumbers>
            {entry.name}: {entry.value}
          </Text>
        ))}
      </VStack>
    </Card>
  )
}

export default function ReinforcementTab() {
  const [episodes, setEpisodes] = useState(400)
  const [alpha, setAlpha] = useState(0.15)
  const [gamma, setGamma] = useState(0.9)
  const [epsilon, setEpsilon] = useState(0.9)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const trainAgent = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/reinforcement/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          episodes,
          alpha,
          gamma,
          epsilon_start: epsilon,
        }),
      })
      const data = await res.json()
      setResult(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    trainAgent()
  }, [])

  const qColumns = result
    ? [
        {
          key: 'segment',
          header: 'Segment',
          width: proportional(1, { minWidth: 140 }),
          renderCell: (row) => <Text weight="semibold">{row.segment}</Text>,
        },
        {
          key: 'churn_risk',
          header: 'Churn Risk',
          width: pixel(105),
          renderCell: (row) => (
            <Text type="supporting" color="secondary">
              {row.churn_risk}
            </Text>
          ),
        },
        {
          key: 'inventory',
          header: 'Inventory',
          width: pixel(95),
          renderCell: (row) => (
            <Text type="supporting" color="secondary">
              {row.inventory}
            </Text>
          ),
        },
        ...result.actions.map((action, idx) => ({
          key: `q_${action.id}`,
          header: action.name,
          width: pixel(115),
          align: 'end',
          renderCell: (row) => (
            <Text
              type="code"
              weight={idx === row.best_action_id ? 'bold' : 'normal'}
              color={idx === row.best_action_id ? 'primary' : 'secondary'}
              hasTabularNumbers
            >
              {row.q_values[idx].toFixed(1)}
            </Text>
          ),
        })),
        {
          key: 'best_action',
          header: 'Optimal Policy',
          width: pixel(175),
          renderCell: (row) => (
            <Badge
              variant={ACTION_BADGE_VARIANTS[row.best_action_id] ?? 'neutral'}
              label={row.best_action}
            />
          ),
        },
      ]
    : []

  return (
    <Layout
      height="fill"
      header={
        <LayoutHeader hasDivider>
          <VStack gap={0.5}>
            <Heading level={1}>Tabular Q-Learning Dynamic Pricing</Heading>
            <Text type="supporting" color="secondary">
              Member 4 · Reinforcement Learning MDP across 24 customer states and 4 discount actions
            </Text>
          </VStack>
        </LayoutHeader>
      }
      start={
        <LayoutPanel
          hasDivider
          width={320}
          role="complementary"
          label="RL Hyperparameters"
        >
          <VStack gap={4}>
            <Slider
              label="Training Episodes"
              min={100}
              max={1000}
              step={50}
              value={episodes}
              onChange={setEpisodes}
              valueDisplay="text"
              formatValue={(v) => `${v} eps`}
            />
            <Slider
              label="Learning Rate (α)"
              min={0.05}
              max={0.5}
              step={0.05}
              value={alpha}
              onChange={setAlpha}
              valueDisplay="text"
              formatValue={(v) => v.toFixed(2)}
            />
            <Slider
              label="Discount Factor (γ)"
              min={0.5}
              max={0.99}
              step={0.01}
              value={gamma}
              onChange={setGamma}
              valueDisplay="text"
              formatValue={(v) => v.toFixed(2)}
            />
            <Slider
              label="Initial Exploration (ε)"
              min={0.1}
              max={1.0}
              step={0.1}
              value={epsilon}
              onChange={setEpsilon}
              valueDisplay="text"
              formatValue={(v) => v.toFixed(1)}
            />

            <Button
              label="Train Q-Agent"
              variant="primary"
              width="100%"
              isLoading={loading}
              onClick={trainAgent}
            />

            <Divider />

            <VStack gap={2}>
              <Heading level={4}>Bellman Update</Heading>
              <CodeBlock
                code={BELLMAN_SNIPPET}
                language="python"
                title="ml/reinforcement.py"
                width="100%"
                hasCopyButton={false}
              />
              <Text type="supporting" color="secondary">
                Pure NumPy tabular Q-learning with decaying epsilon-greedy exploration.
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
                    <Heading level={4}>Learned Policy Profit</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        ${result.benchmark.learned_policy_profit.toLocaleString()}
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary">
                      200-step evaluation
                    </Text>
                  </VStack>
                </Card>

                <Card>
                  <VStack gap={1}>
                    <Heading level={4}>Random Baseline</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        ${result.benchmark.random_policy_profit.toLocaleString()}
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary">
                      Unguided uniform discounting
                    </Text>
                  </VStack>
                </Card>

                <Card>
                  <VStack gap={1}>
                    <Heading level={4}>Profit Lift</Heading>
                    <Heading level={2}>
                      <Text type="inherit" hasTabularNumbers>
                        +{result.benchmark.profit_lift_pct}%
                      </Text>
                    </Heading>
                    <Text type="supporting" color="secondary">
                      Gain over random baseline
                    </Text>
                  </VStack>
                </Card>
              </Grid>

              <Card>
                <VStack gap={3}>
                  <VStack gap={0.5}>
                    <Heading level={3}>Episode Reward Convergence</Heading>
                    <Text type="supporting" color="secondary">
                      Cumulative reward rises as ε decays toward exploitation
                    </Text>
                  </VStack>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={result.reward_curve} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border, rgba(128,128,128,0.2))" />
                      <XAxis
                        dataKey="episode"
                        tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }}
                      />
                      <YAxis tick={{ fontSize: 11, fill: 'var(--color-text-secondary, #8B949E)' }} />
                      <Tooltip content={<ChartTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="reward"
                        name="Episode Reward"
                        stroke="var(--color-data-neutral, #8494A3)"
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="moving_avg"
                        name="Moving Average"
                        stroke="var(--color-data-categorical-orange, #EB6E00)"
                        strokeWidth={3}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </VStack>
              </Card>

              <Card>
                <VStack gap={4}>
                  <VStack gap={0.5}>
                    <Heading level={3}>Learned Q-Table (24 States × 4 Actions)</Heading>
                    <Text type="supporting" color="secondary">
                      Optimal pricing policy learned for every customer state
                    </Text>
                  </VStack>
                  <Table
                    data={result.q_table}
                    columns={qColumns}
                    idKey="state_id"
                    density="compact"
                    dividers="rows"
                    hasHover
                  />
                </VStack>
              </Card>
            </VStack>
          ) : (
            <Card>
              <Text color="secondary">Training tabular Q-Learning agent...</Text>
            </Card>
          )}
        </LayoutContent>
      }
    />
  )
}
