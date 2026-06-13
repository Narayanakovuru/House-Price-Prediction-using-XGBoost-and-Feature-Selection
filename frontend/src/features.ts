export interface SliderField {
  type: 'slider';
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit?: string;
  description?: string;
}

export interface NumberField {
  type: 'number';
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit?: string;
  description?: string;
}

export interface SelectField {
  type: 'select';
  key: string;
  label: string;
  options: { value: string; label: string }[];
  default: string;
  description?: string;
}

export type FormField = SliderField | NumberField | SelectField;

export interface FieldGroup {
  id: string;
  title: string;
  icon: string;
  fields: FormField[];
}

export const FIELD_GROUPS: FieldGroup[] = [
  {
    id: 'quality',
    title: 'Quality & Condition',
    icon: '⭐',
    fields: [
      {
        type: 'slider',
        key: 'OverallQual',
        label: 'Overall Quality',
        min: 1, max: 10, step: 1, default: 6,
        description: 'Rates the overall material and finish quality (1=Very Poor → 10=Excellent)',
      },
      {
        type: 'slider',
        key: 'OverallCond',
        label: 'Overall Condition',
        min: 1, max: 10, step: 1, default: 5,
        description: 'Rates the overall condition of the house',
      },
      {
        type: 'select',
        key: 'ExterQual',
        label: 'Exterior Quality',
        options: [
          { value: 'Ex', label: 'Excellent' },
          { value: 'Gd', label: 'Good' },
          { value: 'TA', label: 'Average' },
          { value: 'Fa', label: 'Fair' },
          { value: 'Po', label: 'Poor' },
        ],
        default: 'TA',
        description: 'Quality of the exterior material',
      },
      {
        type: 'select',
        key: 'KitchenQual',
        label: 'Kitchen Quality',
        options: [
          { value: 'Ex', label: 'Excellent' },
          { value: 'Gd', label: 'Good' },
          { value: 'TA', label: 'Average' },
          { value: 'Fa', label: 'Fair' },
          { value: 'Po', label: 'Poor' },
        ],
        default: 'TA',
        description: 'Kitchen quality rating',
      },
    ],
  },
  {
    id: 'size',
    title: 'Size & Area',
    icon: '📐',
    fields: [
      {
        type: 'number',
        key: 'GrLivArea',
        label: 'Above Ground Living Area',
        min: 300, max: 6000, step: 50, default: 1500,
        unit: 'sq ft',
        description: 'Above grade living area square footage',
      },
      {
        type: 'number',
        key: 'TotalBsmtSF',
        label: 'Total Basement Area',
        min: 0, max: 4000, step: 50, default: 1000,
        unit: 'sq ft',
        description: 'Total basement square footage (0 if none)',
      },
      {
        type: 'number',
        key: '1stFlrSF',
        label: '1st Floor Area',
        min: 300, max: 4000, step: 50, default: 1000,
        unit: 'sq ft',
        description: 'First floor square footage',
      },
      {
        type: 'number',
        key: '2ndFlrSF',
        label: '2nd Floor Area',
        min: 0, max: 2000, step: 50, default: 0,
        unit: 'sq ft',
        description: 'Second floor square footage (0 if none)',
      },
      {
        type: 'number',
        key: 'LotArea',
        label: 'Lot Size',
        min: 1000, max: 50000, step: 500, default: 9000,
        unit: 'sq ft',
        description: 'Total lot area in square feet',
      },
    ],
  },
  {
    id: 'rooms',
    title: 'Rooms & Bathrooms',
    icon: '🛏️',
    fields: [
      {
        type: 'slider',
        key: 'TotRmsAbvGrd',
        label: 'Total Rooms Above Ground',
        min: 2, max: 14, step: 1, default: 6,
        description: 'Total rooms above grade (excludes bathrooms)',
      },
      {
        type: 'slider',
        key: 'FullBath',
        label: 'Full Bathrooms',
        min: 0, max: 4, step: 1, default: 2,
        description: 'Full bathrooms above grade',
      },
      {
        type: 'slider',
        key: 'HalfBath',
        label: 'Half Bathrooms',
        min: 0, max: 2, step: 1, default: 0,
        description: 'Half bathrooms above grade',
      },
      {
        type: 'slider',
        key: 'BedroomAbvGr',
        label: 'Bedrooms',
        min: 0, max: 8, step: 1, default: 3,
        description: 'Bedrooms above basement level',
      },
    ],
  },
  {
    id: 'garage',
    title: 'Garage & Utilities',
    icon: '🚗',
    fields: [
      {
        type: 'slider',
        key: 'GarageCars',
        label: 'Garage Capacity',
        min: 0, max: 5, step: 1, default: 2,
        unit: 'cars',
        description: 'Size of garage in car capacity',
      },
      {
        type: 'number',
        key: 'GarageArea',
        label: 'Garage Area',
        min: 0, max: 1600, step: 50, default: 480,
        unit: 'sq ft',
        description: 'Size of garage in square feet',
      },
      {
        type: 'slider',
        key: 'Fireplaces',
        label: 'Fireplaces',
        min: 0, max: 4, step: 1, default: 1,
        description: 'Number of fireplaces',
      },
    ],
  },
  {
    id: 'age',
    title: 'Age & Remodel',
    icon: '📅',
    fields: [
      {
        type: 'slider',
        key: 'YearBuilt',
        label: 'Year Built',
        min: 1870, max: 2010, step: 1, default: 1973,
        description: 'Original construction year',
      },
      {
        type: 'slider',
        key: 'YearRemodAdd',
        label: 'Year Remodeled',
        min: 1950, max: 2010, step: 1, default: 1994,
        description: 'Remodel year (same as construction if no remodel)',
      },
    ],
  },
  {
    id: 'location',
    title: 'Location',
    icon: '📍',
    fields: [
      {
        type: 'select',
        key: 'Neighborhood',
        label: 'Neighborhood',
        options: [
          { value: 'NAmes', label: 'North Ames' },
          { value: 'CollgCr', label: 'College Creek' },
          { value: 'Crawfor', label: 'Crawford' },
          { value: 'FilMeadow', label: 'Filbert Meadow' },
          { value: 'NoRidge', label: 'Northridge' },
          { value: 'Mitchel', label: 'Mitchell' },
          { value: 'Somerst', label: 'Somerset' },
          { value: 'NWAmes', label: 'Northwest Ames' },
          { value: 'OldTown', label: 'Old Town' },
          { value: 'BrkSide', label: 'Brookside' },
          { value: 'Sawyer', label: 'Sawyer' },
          { value: 'NridgHt', label: 'Northridge Heights' },
          { value: 'Edwards', label: 'Edwards' },
          { value: 'Timber', label: 'Timberland' },
          { value: 'Gilbert', label: 'Gilbert' },
        ],
        default: 'CollgCr',
        description: 'Physical location within Ames city limits',
      },
      {
        type: 'select',
        key: 'MSZoning',
        label: 'Zoning Classification',
        options: [
          { value: 'RL', label: 'Residential Low Density' },
          { value: 'RM', label: 'Residential Medium Density' },
          { value: 'FV', label: 'Floating Village Residential' },
          { value: 'RH', label: 'Residential High Density' },
          { value: 'C (all)', label: 'Commercial' },
        ],
        default: 'RL',
        description: 'General zoning classification of the sale',
      },
    ],
  },
];

export function buildDefaultValues(): Record<string, number | string> {
  const values: Record<string, number | string> = {};
  for (const group of FIELD_GROUPS) {
    for (const field of group.fields) {
      values[field.key] = field.default;
    }
  }
  return values;
}
