import React from 'react';

import './CourseGoals.css';
import Card from '../UI/Card';
import GoalItem from './GoalItem';

function CourseGoals(props) {
  const hasNoGoals = !props.goals || props.goals.length === 0;

  return (
    <section id='course-goals'>
      <Card>
        <h2>잘 반영되나요?</h2>
        {hasNoGoals && <h2>No goals found. Start adding some!</h2>}
        <ul>
          {props.goals.map((goal) => (
            <GoalItem
              key={goal.id}
              id={goal.id}
              text={goal.text}
              onDelete={props.onDeleteGoal}
            />
          ))}
        </ul>
      </Card>
    </section>
  );
}

export default CourseGoals;
