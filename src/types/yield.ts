export interface ProcessRun {
  id: string;
  created_at: string;
  recipe_name: string;
  etching_time: number; // in seconds
  pressure: number; // in mTorr
  temperature: number; // in Celsius
  gas_flow_rate: number; // in sccm
}

export interface YieldResult {
  id: string;
  process_run_id: string;
  yield_rate: number; // percentage (0-100)
  efficiency: number; // normalized efficiency (0-1)
  defect_density: number; // counts/cm2
  measured_at: string;
}

export type ProcessWithYield = ProcessRun & {
  yield_results?: YieldResult[];
};
