import { useEffect, useState } from 'react'
import { Theme } from '@astryxdesign/core/theme'
import { AppShell } from '@astryxdesign/core/AppShell'
import {
  SideNav,
  SideNavHeading,
  SideNavItem,
  SideNavSection,
} from '@astryxdesign/core/SideNav'
import { Card } from '@astryxdesign/core/Card'
import { NavIcon } from '@astryxdesign/core/NavIcon'
import { Badge } from '@astryxdesign/core/Badge'
import { Divider } from '@astryxdesign/core/Divider'
import { HStack, VStack } from '@astryxdesign/core/Layout'
import { StatusDot } from '@astryxdesign/core/StatusDot'
import { Text } from '@astryxdesign/core/Text'
import { Icon } from '@astryxdesign/core/Icon'
import { themes, defaultThemeSlug } from './astryx-themes'
import DatasetView from './components/DatasetView'
import UnsupervisedTab from './components/UnsupervisedTab'
import SupervisedTab from './components/SupervisedTab'
import ReinforcementTab from './components/ReinforcementTab'

const TABS = [
  { id: 'dataset', label: 'Data Pipeline', code: 'M1', icon: 'viewColumns' },
  { id: 'unsupervised', label: 'Unsupervised', code: 'M2', icon: 'funnel' },
  { id: 'supervised', label: 'Supervised', code: 'M3', icon: 'wrench' },
  { id: 'reinforcement', label: 'Reinforcement', code: 'M4', icon: 'arrowUp' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('dataset')
  const [dataset, setDataset] = useState(null)
  const [backendOnline, setBackendOnline] = useState(false)

  const loadDataset = async () => {
    try {
      const res = await fetch('/api/dataset')
      if (res.ok) {
        const data = await res.json()
        setDataset(data)
        setBackendOnline(true)
      }
    } catch {
      setBackendOnline(false)
    }
  }

  useEffect(() => {
    loadDataset()
  }, [])

  const sideNav = (
    <SideNav
      aria-label="ML Modules Navigation"
      header={
        <Card padding={3} variant="muted">
          <VStack gap={2}>
            <SideNavHeading
              superheading="IS FINAL LAB"
              heading="ShopSense AI"
              subheading="3-Paradigm ML Suite"
              icon={<NavIcon icon={<Icon icon="viewColumns" size="sm" />} />}
              headerEndContent={<Badge label="v2.4" variant="blue" />}
            />
            <Divider />
            <HStack gap={1} vAlign="center" wrap="wrap">
              <Badge label="K-Means" variant="cyan" />
              <Badge label="Churn RF" variant="purple" />
              <Badge label="Q-Agent" variant="teal" />
            </HStack>
          </VStack>
        </Card>
      }
      footer={
        <VStack gap={1} padding={3}>
          <HStack gap={2} vAlign="center">
            <StatusDot
              variant={backendOnline ? 'success' : 'error'}
              label={backendOnline ? 'FastAPI Connected' : 'Backend Offline'}
            />
            <Text type="supporting">
              {backendOnline ? 'FastAPI Connected' : 'Backend Offline'}
            </Text>
          </HStack>
          <Text type="supporting" color="secondary" hasTabularNumbers>
            {dataset ? `${dataset.rows.toLocaleString()} Customer Rows` : 'Waiting for data...'}
          </Text>
        </VStack>
      }
    >
      <SideNavSection title="ML Modules">
        {TABS.map((tab) => (
          <SideNavItem
            key={tab.id}
            label={tab.label}
            icon={tab.icon}
            isSelected={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
            endContent={
              <Text type="supporting" color="secondary" hasTabularNumbers>
                {tab.code}
              </Text>
            }
          />
        ))}
      </SideNavSection>
    </SideNav>
  )

  return (
    <Theme theme={themes[defaultThemeSlug]} mode="dark">
      <AppShell height="fill" sideNav={sideNav}>
        {activeTab === 'dataset' && (
          <DatasetView dataset={dataset} onDatasetUpdate={setDataset} />
        )}
        {activeTab === 'unsupervised' && <UnsupervisedTab />}
        {activeTab === 'supervised' && <SupervisedTab />}
        {activeTab === 'reinforcement' && <ReinforcementTab />}
      </AppShell>
    </Theme>
  )
}
