export type KpiCardProps = {
  label: string;
  value: string;
  sub?: string;
};

export function KpiCard({ label, value, sub }: KpiCardProps) {
  return (
    <div className="models-kpi">
      <div className="models-kpi-label">{label}</div>
      <div className="models-kpi-value">{value}</div>
      {sub && <div className="models-kpi-sub">{sub}</div>}
    </div>
  );
}
