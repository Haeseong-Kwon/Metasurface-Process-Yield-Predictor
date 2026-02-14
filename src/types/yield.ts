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

export interface Defect {
  id: string;
  type: 'Broken' | 'Bridge' | 'Short' | 'Particle';
  x: number; // normalized 0-1
  y: number; // normalized 0-1
  width: number;
  height: number;
}

export interface OptimizationAdvice {
  id: string;
  category: 'Etching' | 'Dose' | 'Flow' | 'Pressure';
  suggestion: string;
  impact_percent: number;
  priority: 'High' | 'Medium' | 'Low';
}

export type ProcessWithYield = ProcessRun & {
  yield_results?: YieldResult[];
  defects?: Defect[];
  advice?: OptimizationAdvice[];
  sem_image_url?: string;
};

export interface YieldPrediction {
  id?: string;
  e_beam_dose: number;
  etching_time: number;
  gas_flow: number;
  pressure: number;
  predicted_yield: number;
  risk_factors: string[];
  created_at?: string;
}
